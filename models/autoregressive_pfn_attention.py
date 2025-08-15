#%%
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
from jaxtyping import Float, Array
from dataclasses import dataclass
import numpy as np

from model_config import ModelConfig

device = t.device("cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu")

"""
A PFN trained to perform linear regression autoregressively. Note the attention mask is slightly different from the original paper in order to allow for autoregressive predictions.
"""


#%%

class PFNAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def forward(
        self, normalized_resid_pre: Float[Array, "batch posn d_model"]
    ) -> Float[Array, "batch posn d_model"]:
        # Calculate query, key and value vectors
        q = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_V
        )

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q,
            k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head**0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = einops.einsum(
            v,
            attn_pattern,
            "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head",
        )

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = (
            einops.einsum(
                z,
                self.W_O,
                "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
            )
            + self.b_O
        )

        return attn_out

    def apply_causal_mask(
        self, attn_scores: Float[Array, "batch n_heads query_pos key_pos"]
    ) -> Float[Array, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores that only allows attention to previous even positions.
        Also allows each position to attend to itself.
        """
        # Get dimensions
        query_pos = attn_scores.size(-2)
        key_pos = attn_scores.size(-1)
        device = attn_scores.device
        
        # Create base causal mask (lower triangular without diagonal)
        lower_no_diag = t.tril(t.ones(query_pos, key_pos, device=device), diagonal=-1)
        
        # Create mask for even key positions
        even_key_positions = (t.arange(key_pos, device=device) % 2 - 1 == 0).float() #need to subtract 1 to make it 0,1,0,1,0,1, etc.
 
        # Combine masks - only allow attention to previous even positions
        allowed = lower_no_diag * even_key_positions
        
        # Allow each position to attend to itself
        allowed.diagonal().fill_(1.0)
        
        # Convert to boolean mask of positions to ignore
        mask = ~allowed.bool()
        
        # Apply the mask to attention scores
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


# %%

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Quick test for PFNAttention's causal mask
    cfg = ModelConfig()
    attn = PFNAttention(cfg).to(device)

    # Create dummy attention scores for easy inspection
    batch_size = 1
    n_heads = cfg.n_heads
    query_pos = 10
    key_pos = 10
    dummy_scores = t.zeros((batch_size, n_heads, query_pos, key_pos), device=device)

    masked_scores = attn.apply_causal_mask(dummy_scores.clone())

    # Check mask pattern for first head
    mask_pattern = (masked_scores[0,0] != attn.IGNORE).int().cpu().numpy()  # 1 = allowed, 0 = blocked
    print("Allowed positions (1=allowed, 0=blocked):")
    print(mask_pattern)
    plt.imshow(mask_pattern, cmap="gray")
    plt.title("Allowed Attention (White = allowed)")
    plt.show()

# %%
