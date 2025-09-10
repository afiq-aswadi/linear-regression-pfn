#%%
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from jaxtyping import Float, Array

from dataclasses import dataclass
from .pfn_embedding import JointEmbedding, Unembed
from .autoregressive_pfn_attention import PFNAttention

from .model_config import ModelConfig

#%%   

class AutoregressivePFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = JointEmbedding(cfg)
        self.transformer = Transformer(cfg)

    def forward(self, x: Float[Array, "batch n_x d_x"], y: Float[Array, "batch n_y d_y"]) -> Float[Array, "batch position d_vocab"]:
        assert x.shape[0] == y.shape[0], "x and y must have the same batch size" # this can probably broadcast but wtv
        assert x.shape[1] == y.shape[1], "x and y must have the same number of points"
        x_seq, y_seq = construct_sequence(x, y)
        residual = self.embed(x_seq, y_seq)
        return self.transformer(residual)
    
    def get_y_embedding(self, y: Float[Array, "batch n_y d_y"]) -> Float[Array, "batch n_y d_model"]:
        return self.embed.get_y_embedding(y)
    
    def get_x_embedding(self, x: Float[Array, "batch n_x d_x"]) -> Float[Array, "batch n_x d_model"]:
        return self.embed.get_x_embedding(x)

    def save(self, path: str):
        t.save(self.state_dict(), path)

    def get_model_mean_prediction(self, x: Float[Array, "batch n_x d_x"], y: Float[Array, "batch n_y d_y"]) -> tuple[Float[Array, "batch n_y d_vocab"], Float[Array, "batch n_y"]]: #todo: better name for function?
        """
        Get the model's mean prediction for a given x and y.

        Returns:   
            probs: Float[Array, "batch n_y d_vocab"]
            model_mean: Float[Array, "batch n_y"]
        """
        # Use bin centers instead of linspace endpoints
        bin_width = (self.cfg.y_max - self.cfg.y_min) / self.cfg.d_vocab
        bin_centers = t.linspace(
            self.cfg.y_min + bin_width/2, 
            self.cfg.y_max - bin_width/2, 
            self.cfg.d_vocab, 
            device=x.device, dtype=x.dtype
        )
        with t.no_grad():
            logits = self.forward(x,y) 
            probs = F.softmax(logits, dim=-1)
            model_mean = t.sum(bin_centers * probs, dim=-1).detach()
            model_mean = model_mean[:, ::2] #even indices
        
        return probs, model_mean
    
    @classmethod
    def load(cls, path: str):
        model = cls(ModelConfig())
        model.load_state_dict(t.load(path))
        return model

def construct_sequence(x: Float[Array, "batch n_x d_x"], y: Float[Array, "batch n_y d_y"]) -> tuple[Float[Array, "batch position d_x"], Float[Array, "batch position d_y"]]:
    """
    Constructs autoregressive sequence: (x_1,0), (x_1,y_1), (x_2,0), (x_2,y_2), ...
    Returns x_seq and y_seq where model trains only on odd indices.
    """
    batch_size, n_points, d_x = x.shape
    _, _, d_y = y.shape
    
    # Create sequence of length 2 * n_points
    x_seq = t.zeros(batch_size, 2 * n_points, d_x, device=x.device, dtype=x.dtype)
    y_seq = t.zeros(batch_size, 2 * n_points, d_y, device=y.device, dtype=y.dtype)
    
    # Fill even indices: (x_i, 0)
    x_seq[:, ::2] = x  # x_1, x_2, x_3, ...
    y_seq[:, ::2] = 0  # 0, 0, 0, ...
    
    # Fill odd indices: (x_i, y_i) 
    x_seq[:, 1::2] = x  # x_1, x_2, x_3, ...
    y_seq[:, 1::2] = y  # y_1, y_2, y_3, ...
    
    return x_seq, y_seq

def bin_y_values(y: t.Tensor, y_min: float = -3.0, y_max: float = 3.0, n_bins: int = 128) -> t.Tensor:
    """
    Convert continuous y values to discrete bin indices.
    
    Args:
        y: continuous y values of shape [batch, num_examples, d_y] 
        y_min, y_max: range for binning
        n_bins: number of bins (should match d_vocab in config)
    
    Returns:
        bin_indices: discrete indices of shape [batch, num_examples]
    """
    # Clamp values to be within range
    y_clamped = t.clamp(y.squeeze(-1), y_min, y_max)
    
    # Convert to bin indices
    bin_width = (y_max - y_min) / n_bins
    bin_indices = ((y_clamped - y_min) / bin_width).long()
    bin_indices = t.clamp(bin_indices, 0, n_bins - 1)
    
    return bin_indices

def unbin_y_values(bin_indices: t.Tensor, y_min: float = -3.0, y_max: float = 3.0, n_bins: int = 128) -> t.Tensor:
    """
    Convert discrete bin indices back to continuous y values.
    
    Args:
        bin_indices: discrete indices of shape [batch, num_examples]
        y_min, y_max: range for binning  
        n_bins: number of bins
        
    Returns:
        y: continuous y values of shape [batch, num_examples, 1]
    """
    bin_width = (y_max - y_min) / n_bins
    y_continuous = y_min + (bin_indices.float() + 0.5) * bin_width  # Use bin centers
    return y_continuous.unsqueeze(-1)


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([AutoregressivePFNBlock(cfg) for _ in range(cfg.n_layers)])
        self.unembed = Unembed(cfg)
    
    def forward(self, residual: Float[Array, "batch position d_model"]) -> Float[Array, "batch position d_vocab"]:
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(residual)
        return logits
    
class AutoregressivePFNBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.attention = PFNAttention(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.GELU(),
            nn.Linear(cfg.d_mlp, cfg.d_model))

    def forward(self, resid_pre: Float[Array, "batch position d_model"]) -> Float[Array, "batch position d_model"]:
        resid_mid = self.attention(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post
    
# %%

if __name__ == "__main__":
    model = AutoregressivePFN(ModelConfig())
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

# %%
