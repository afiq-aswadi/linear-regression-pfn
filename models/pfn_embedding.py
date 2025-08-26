#%%
import torch.nn as nn
from .model_config import ModelConfig
from jaxtyping import Float, Array

class JointEmbedding(nn.Module):
    """ 
    Gets joint embedding of x and y by summing the embeddings of x and y.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.x_embedding = XEmbedding(cfg)
        self.y_embedding = YEmbedding(cfg)

    def forward(self, x: Float[Array, "batch position d_x"], y: Float[Array, "batch position d_y"]): #note: some papers use a concatentation of the embeddings instead of a sum, but the original pfn paper uses a sum.
        return self.x_embedding(x) + self.y_embedding(y)
    
    def get_y_embedding(self, y: Float[Array, "batch position d_y"]) -> Float[Array, "batch position d_model"]:
        return self.y_embedding(y) #returns the embedding of y
    
    def get_x_embedding(self, x: Float[Array, "batch position d_x"]) -> Float[Array, "batch position d_model"]:
        return self.x_embedding(x) #returns the embedding of x

class XEmbedding(nn.Module): 
    """
    Embeds x values into embedding space.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embedding = nn.Linear(cfg.d_x, cfg.d_model)

    def forward(self, x: Float[Array, "batch position d_x"]) -> Float[Array, "batch position d_model"]:
        return self.embedding(x)

class YEmbedding(nn.Module):
    """
    Embeds y values into embedding space. Typically y is a scalar, so input_dim = 1. Note these are continuous embeddings rather than discrete like in language models.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embedding = nn.Linear(cfg.d_y, cfg.d_model)

    def forward(self, y: Float[Array, "batch position d_y"]) -> Float[Array, "batch position d_model"]:
        return self.embedding(y)


class Unembed(nn.Module):
    """
    Unembeds the model output into the y space. We only train on even indices.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.unembedding = nn.Linear(cfg.d_model, cfg.d_vocab)

    def forward(self, resid_post: Float[Array, "batch position d_model"]) -> Float[Array, "batch position d_vocab"]:
        return self.unembedding(resid_post)
# %%
