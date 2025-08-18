import torch
import numpy as np
from models.model_config import ModelConfig
from models.model import unbin_y_values

# ---------------------------------------------------------------
# Predictive resampling
# ---------------------------------------------------------------
@torch.no_grad()
def predictive_resampling_beta(
    model: torch.nn.Module,
    config: ModelConfig,
    forward_recursion_steps: int = 128,        # total trajectory length T
    forward_recursion_samples: int = 1000,     # B for this chunk
    sample_y: bool = True,
    init_x: torch.Tensor = None,  # optional prefix buffer (batch of 1 or None)
    init_y: torch.Tensor = None,  # optional prefix buffer (batch of 1 or None)
) -> np.ndarray:
    """
    If init_tokens is None: behavior with fresh X ~ N(0,I).
    If init_tokens is a batch of 1: replicate it to forward_recursion_samples (B).

    If init_tokens is not None, then it must be a compatible input to the model.
    """
    device = next(model.parameters()).device
    D = config.d_x #input dimension

    B = forward_recursion_samples # Batch size for this chunk
    T = forward_recursion_steps

    # 1) Build the full token buffer
    if init_x is None:
        # no prefix: sample all X at once
        X = torch.randn(B, T, D, device=device)
        K_init = 0
        # Initialize y with a single zero value to start
        y = torch.zeros(B, 1, config.d_y, device=device)
    else:
        assert init_x.shape[0] == 1, "init_x must have batch size 1"
        assert init_x.shape[1] == init_y.shape[1], "init_x and init_y must have the same length"

        x_prefix = init_x.repeat(B, 1, 1).to(device) # Replicate for the batch size B
        y_prefix = init_y.repeat(B, 1, 1).to(device) # Replicate for the batch size B

        K_init = x_prefix.shape[1] 

        X_fut = torch.randn(B, T - K_init, D, device=device)
        X = torch.cat([x_prefix, X_fut], dim=1)  # (B, T, D)
        y = y_prefix

    for k in range(K_init, T-1):
        ctx_x = X[:,:k+1,:]
        
        # ctx_y should contain all the y values we have so far, 
        # plus zero padding if we need more to match ctx_x length
        current_y_len = y.shape[1]
        needed_len = k + 1
        
        if current_y_len >= needed_len:
            ctx_y = y[:,:needed_len,:]
        else:
            # Pad with zeros to match the required length
            pad_len = needed_len - current_y_len
            assert pad_len == 1, "wtf? pad_len is not 1"
            pad_tensor = torch.zeros(B, pad_len, config.d_y, device=device)
            ctx_y = torch.cat([y, pad_tensor], dim=1)
            
        assert ctx_x.shape[1] == ctx_y.shape[1], "ctx_x and ctx_y must have the same length"

        model_output = model(ctx_x, ctx_y)
        # The real model should return sequences of length 2*input_length
        # We want the second-to-last position which corresponds to (x_{k+1}, 0)
        if model_output.shape[1] >= 2:
            logits = model_output[:, -2, :]
        else:
            # Fallback for edge cases or mock models
            logits = model_output[:, -1, :]
        probs  = torch.softmax(logits, dim=-1)
        if sample_y:
            y_cls = torch.multinomial(probs, 1).squeeze(-1)
            y_sample = unbin_y_values(y_cls.unsqueeze(-1), config.y_min, config.y_max, config.d_vocab)
        else:
            y_cls = torch.argmax(probs, dim=-1)
            y_sample = unbin_y_values(y_cls.unsqueeze(-1), config.y_min, config.y_max, config.d_vocab)

        y = torch.cat([y, y_sample], dim=1)

    # # 3) sample y for steps k = K_init … T-1
    # for k in range(K_init, T - 1):
    #     ctx = tokens[:, : (2*k + 1), :]           # up to x_k
    #     h = model.in_proj(ctx) + model._get_pos_emb(2*k+1, device, ctx.dtype)
    #     h = model.tr(h, mask=causal_mask[:2*k+1, :2*k+1])
    #     logits = model.readout(h[:, -1, :])       # (B, NUM_BINS)
    #     probs  = torch.softmax(logits, dim=-1)

    #     if sample_y:
    #         y_cls = torch.multinomial(probs, 1).squeeze(-1)
    #     else:
    #         y_cls = torch.argmax(probs, dim=-1)

    #     tokens[:, 2*k + 1, 0] = y_cls.float()

        if k % 20 == 0 or k == T - 2:
            # Class ID diversity diagnostic
            print(f"[Step {k}] Sampled class IDs:")
            print(torch.unique(y_cls, return_counts=True))

            # Entropy diagnostic
            entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
            print(f"[Step {k}] Average predictive entropy: {entropy:.2f}")


    # 4) final y_T prediction
    final_model_output = model(X, y)
    if final_model_output.shape[1] >= 2:
        final_logits = final_model_output[:, -2, :]
    else:
        # Fallback for edge cases
        final_logits = final_model_output[:, -1, :]
    yT_cls = torch.argmax(final_logits, dim=-1) #why we argmax here?
    yT_real = unbin_y_values(yT_cls.unsqueeze(-1), config.y_min, config.y_max, config.d_vocab)
    y = torch.cat([y, yT_real], dim=1)

    # Remove the initial padding zero when there was no init
    if K_init == 0:
        y = y[:, 1:, :]  # Remove the first element (the initial zero)
    try:
        beta_hat = torch.linalg.lstsq(X, y).solution.squeeze(-1)  # (B, D)
    except Exception as e:
        print(f"Error in lstsq: {e}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        # Manual implementation of least squares: β = (X^T X)^(-1) X^T y
        X_t = X.transpose(-2, -1)  # [B, D, T]
        X_t_X = torch.bmm(X_t, X)  # [B, D, D]
        X_t_y = torch.bmm(X_t, y)  # [B, D, 1]
        try:
            # Try using inverse first
            X_t_X_inv = torch.linalg.inv(X_t_X)  # [B, D, D]
            beta_hat = torch.bmm(X_t_X_inv, X_t_y).squeeze(-1)  # [B, D]
        except Exception:
            # If inverse fails, try using pinverse
            print("Inverse failed, trying pseudoinverse")
            X_pinv = torch.linalg.pinv(X)  # [B, D, T]
            beta_hat = torch.bmm(X_pinv, y).squeeze(-1)  # [B, D]

    return beta_hat.cpu().numpy()

@torch.no_grad()
def predictive_resampling_beta_chunked(
    model: torch.nn.Module,
    config: ModelConfig,
    forward_recursion_steps: int,
    forward_recursion_samples: int, # B_total
    chunk_size: int = 200, # B_chunk
    sample_y: bool = True,
    init_x: torch.Tensor = None, # Assumed batch of 1 or None
    init_y: torch.Tensor = None, # Assumed batch of 1 or None
) -> np.ndarray:
    """
    Run predictive_resampling_beta in chunks to avoid OOM.
    init_tokens (if not None) is a batch of size 1, which will be replicated
    by predictive_resampling_beta for each chunk.
    """
    all_betas = []
    total_samples = forward_recursion_samples

    # Check if init_tokens has batch size 1 if provided (optional safety check)
    if init_x is not None:
        if init_x.shape[0] != 1:
            raise ValueError(f"init_x batch size ({init_x.shape[0]}) must be 1 when provided.")
        assert init_x.shape[1] == init_y.shape[1], "init_x and init_y must have the same length"

    for start in range(0, total_samples, chunk_size):
        this_chunk = min(chunk_size, total_samples - start)

        betas_chunk = predictive_resampling_beta(
            model,
            config,
            forward_recursion_steps=forward_recursion_steps,
            # Pass the chunk size as the number of samples for this batch (B for predictive_resampling_beta)
            forward_recursion_samples=this_chunk,
            sample_y=sample_y,
            init_x=init_x, # Pass the batch of 1 (or None) received
            init_y=init_y, # Pass the batch of 1 (or None) received
        )
        all_betas.append(betas_chunk)

    return np.concatenate(all_betas, axis=0)