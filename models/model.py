#%%
import torch as t
import torch.nn as nn
from jaxtyping import Float, Array

from dataclasses import dataclass
from pfn_embedding import JointEmbedding, Unembed
from autoregressive_pfn_attention import PFNAttention

from model_config import ModelConfig

#%%   

class AutoregressivePFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = JointEmbedding(cfg)
        self.transformer = Transformer(cfg)

    def forward(self, x: Float[Array, "batch n_x d_x"], y: Float[Array, "batch n_y d_y"]) -> Float[Array, "batch position d_vocab"]:
        residual = self.embed(x, y)
        return self.transformer(residual)
    
    def get_y_embedding(self, y: Float[Array, "batch n_y d_y"]) -> Float[Array, "batch n_y d_model"]:
        return self.embed.get_y_embedding(y)
    
    def get_x_embedding(self, x: Float[Array, "batch n_x d_x"]) -> Float[Array, "batch n_x d_model"]:
        return self.embed.get_x_embedding(x)

def construct_sequence(x: Float[Array, "batch n_x d_x"], y: Float[Array, "batch n_y d_y"]) -> Float[Array, "batch position d_model"]:
    pass




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
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post
    
# %%

if __name__ == "__main__":
    model = AutoregressivePFN(ModelConfig())
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

# %%
