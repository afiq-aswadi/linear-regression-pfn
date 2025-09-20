import torch
import numpy as np
from models.config import ModelConfig
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
    save_y: bool = False,
    debug: bool = False,
) -> np.ndarray:
    """
    If init_tokens is None: behavior with fresh X ~ N(0,I).
    If init_tokens is a batch of 1: replicate it to forward_recursion_samples (B).

    If init_tokens is not None, then it must be a compatible input to the model.

    If save_y is True, then return the beta_hat and the y trajectory.
    If save_y is False, then return the beta_hat.
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

        X_fut = torch.randn(B, T , D, device=device)
        X = torch.cat([x_prefix, X_fut], dim=1)  # (B, T + K_init, D)
        y = y_prefix # (B, K_init, d_y)
        y = torch.cat([y, torch.zeros(B, 1, config.d_y, device=device)], dim=1)  # Append zero for next prediction

    for k in range(K_init, T+K_init): #TODO: check potential off by one error
        ctx_x = X[:, :k+1, :]

        # Ensure y has a zero placeholder appended for the current step,
        # then pass that as context and replace the placeholder with the sample.
        current_y_len = y.shape[1]
        needed_len = k + 1

        if current_y_len < needed_len:
            pad_len = needed_len - current_y_len
            assert pad_len == 1, "expected to pad exactly one step"
            pad_tensor = torch.zeros(B, pad_len, config.d_y, device=device)
            y = torch.cat([y, pad_tensor], dim=1)
        elif current_y_len > needed_len:
            # Should not happen; truncate defensively
            y = y[:, :needed_len, :]

        ctx_y = y[:, :needed_len, :]
        assert ctx_x.shape[1] == ctx_y.shape[1], "ctx_x and ctx_y must have the same length"

        model_output = model(ctx_x, ctx_y)
        logits = model_output[:, -2, :]
        probs  = torch.softmax(logits, dim=-1)
        if sample_y:
            y_cls = torch.multinomial(probs, 1).squeeze(-1)
            y_sample = unbin_y_values(y_cls.unsqueeze(-1), config.y_min, config.y_max, config.d_vocab)
        else:
            y_cls = torch.argmax(probs, dim=-1)
            y_sample = unbin_y_values(y_cls.unsqueeze(-1), config.y_min, config.y_max, config.d_vocab)

        # Replace the last zero placeholder with the sampled/argmax value
        y[:, -1:, :] = y_sample


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

        if k % 20 == 0 and k == T - 2 and debug:
            # Class ID diversity diagnostic
            print(f"[Step {k}] Sampled class IDs:")
            print(torch.unique(y_cls, return_counts=True))

            # Entropy diagnostic
            entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
            print(f"[Step {k}] Average predictive entropy: {entropy:.2f}")

    # 4) final y_T prediction: append a zero placeholder then replace it
    if y.shape[1] < T + K_init: 
        pad_len = T + K_init - y.shape[1]
        assert pad_len == 1, "expected to pad exactly one step at final prediction"
        y = torch.cat([y, torch.zeros(B, pad_len, config.d_y, device=device)], dim=1)
    # elif y.shape[1] > T:
        # y = y[:, :T, :]
    assert X.shape[1] == y.shape[1], f"X and y must have the same length, currently X.shape[1] = {X.shape[1]}, y.shape[1] = {y.shape[1]}"

    final_model_output = model(X, y)
    if final_model_output.shape[1] >= 2:
        final_logits = final_model_output[:, -2, :]
    else:
        final_logits = final_model_output[:, -1, :]
    final_probs = torch.softmax(final_logits, dim=-1)
    if sample_y:
        yT_cls = torch.multinomial(final_probs, 1).squeeze(-1)
    else:
        yT_cls = torch.argmax(final_probs, dim=-1)
    yT_real = unbin_y_values(yT_cls.unsqueeze(-1), config.y_min, config.y_max, config.d_vocab)
    y[:, -1:, :] = yT_real
    try:
        beta_hat = torch.linalg.lstsq(X, y).solution.squeeze(-1)  # (B, D)
    except Exception as e:
        print("Error in lstsq, trying manual implementation (mps does not support lstsq)")
        # print(f"Error in lstsq: {e}")
        # print(f"X shape: {X.shape}, y shape: {y.shape}")
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

    if save_y:
        return beta_hat.cpu().numpy(), y.cpu().numpy()
    else:
        return beta_hat.cpu().numpy()

@torch.no_grad()
def predictive_resampling_beta_chunked(
    model: torch.nn.Module,
    config: ModelConfig,
    forward_recursion_steps: int,
    forward_recursion_samples: int, # B_total
    chunk_size: int = 200, # B_chunk
    sample_y: bool = True,
    save_y: bool = False,
    init_x: torch.Tensor = None, # Assumed batch of 1 or None
    init_y: torch.Tensor = None, # Assumed batch of 1 or None
    debug: bool = False,
) -> np.ndarray:
    """
    Run predictive_resampling_beta in chunks to avoid OOM.
    init_tokens (if not None) is a batch of size 1, which will be replicated
    by predictive_resampling_beta for each chunk.
    """
    all_betas = []
    all_ys = [] if save_y else None
    total_samples = forward_recursion_samples

    # Check if init_tokens has batch size 1 if provided (optional safety check)
    if init_x is not None:
        if init_x.shape[0] != 1:
            raise ValueError(f"init_x batch size ({init_x.shape[0]}) must be 1 when provided.")
        assert init_x.shape[1] == init_y.shape[1], "init_x and init_y must have the same length"

    for start in range(0, total_samples, chunk_size):
        this_chunk = min(chunk_size, total_samples - start)

        result = predictive_resampling_beta(
            model,
            config,
            forward_recursion_steps=forward_recursion_steps,
            # Pass the chunk size as the number of samples for this batch (B for predictive_resampling_beta)
            forward_recursion_samples=this_chunk,
            sample_y=sample_y,
            init_x=init_x, # Pass the batch of 1 (or None) received
            init_y=init_y, # Pass the batch of 1 (or None) received
            save_y=save_y,
            debug=debug,
        )
        
        if save_y:
            betas_chunk, y_chunk = result
            all_betas.append(betas_chunk)
            all_ys.append(y_chunk)
        else:
            all_betas.append(result)

    if save_y:
        return np.concatenate(all_betas, axis=0), np.concatenate(all_ys, axis=0)
    else:
        return np.concatenate(all_betas, axis=0)