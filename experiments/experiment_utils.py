"""
Shared utilities for experiment scripts:
- Device selection
- Default model config factory
- Checkpoint directory and path helpers
- Model loading from checkpoints
- Task distribution loading helpers
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from models.model import AutoregressivePFN, bin_y_values 
from models.model_config import ModelConfig
from samplers.tasks import load_task_distribution_from_pt as _load_td


def get_device(prefer: str = "cuda") -> str:
    """
    Return the best available device string: "cuda" > "mps" > "cpu".
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_checkpoints_dir() -> str:
    """
    Resolve checkpoints directory, preferring CHECKPOINTS_DIR env var, otherwise
    experiments/checkpoints relative to this file.
    """
    env = os.environ.get("CHECKPOINTS_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(__file__), "checkpoints")


def build_checkpoint_path(checkpoints_dir: str, run_id: str, num_tasks, ckpt_idx: int) -> str:
    return os.path.join(checkpoints_dir, f"{run_id}_model_{num_tasks}tasks_step_{ckpt_idx}.pt") #todo: make this better


def get_pretrain_distribution_path(checkpoints_dir: str, run_id: str, num_tasks: int, task_size: int) -> str:
    return os.path.join(checkpoints_dir, f"{run_id}_pretrain_discrete_{num_tasks}tasks_{task_size}d.pt")


def get_true_distribution_path(checkpoints_dir: str, run_id: str, task_size: int) -> str:
    return os.path.join(checkpoints_dir, f"{run_id}_true_gaussian_{task_size}d.pt")


def load_model_from_checkpoint(config: ModelConfig, checkpoint_path: str, device: Optional[str] = None) -> AutoregressivePFN:
    """
    Create an AutoregressivePFN with the provided config and load weights from checkpoint.
    """
    if device is None:
        device = get_device()
    model = AutoregressivePFN(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    # model = torch.compile(model) ## sometimes this breaks?
    return model


def load_task_distribution(file_path: str, device: Optional[str] = None):
    if device is None:
        device = get_device()
    return _load_td(file_path, device=device)


def extract_w_pool(task_distribution) -> Optional[np.ndarray]:
    """
    Extract task pool (weights) as a numpy array if present in a loaded distribution.
    Returns None if unavailable.
    """
    w_pool = None
    if isinstance(task_distribution, dict) and "tasks" in task_distribution:
        w_pool = task_distribution["tasks"]
    elif hasattr(task_distribution, "tasks"):
        w_pool = getattr(task_distribution, "tasks")
    else:
        # Assume the object itself may be an array-like pool
        w_pool = task_distribution

    if isinstance(w_pool, torch.Tensor):
        return w_pool.detach().cpu().numpy()
    if isinstance(w_pool, np.ndarray):
        return w_pool
    return None



def ridge_ppd(xs, ys, model_config: ModelConfig, sigma_squared=0.25):
    """
    Compute Ridge regression posterior predictive distributions autoregressively.
    For each position i, predict y_i using context from positions 0 to i-1.
    
    Args:
        xs: Input points, shape (batch_size, n_context, d_x)
        ys: Output values, shape (batch_size, n_context) 
        model_config: Model configuration containing bucket parameters
        sigma_squared: Noise variance (default 0.25)
    
    Returns:
        discrete_probs: Discrete probabilities over buckets for each position, 
                       shape (batch_size, n_context, d_vocab)
    """
    device = xs.device if hasattr(xs, 'device') else 'cpu'
    
    # Convert to numpy for scipy operations
    if torch.is_tensor(xs):
        xs_np = xs.detach().cpu().numpy()
        ys_np = ys.detach().cpu().numpy()
    else:
        xs_np = xs
        ys_np = ys
    
    batch_size, n_context, d_x = xs_np.shape
    discrete_probs = np.zeros((batch_size, n_context, model_config.d_vocab))
    
    # Setup bucket parameters
    bucket_edges = np.linspace(model_config.y_min, model_config.y_max, model_config.d_vocab + 1)
    
    for b in range(batch_size):
        for i in range(n_context):
            if i == 0:
                # First prediction: no context, use prior N(0, σ²)
                mean_i = 0.0
                var_i = sigma_squared
            else:
                # Use context from positions 0 to i-1
                x_context = xs_np[b, :i]  # shape (i, d_x)
                y_context = ys_np[b, :i]  # shape (i,)
                x_star = xs_np[b, i:i+1]  # shape (1, d_x)
                
                # Compute Ridge regression posterior parameters
                # μ = (X^T X + σ²I)^{-1} X^T y
                XTX = x_context.T @ x_context
                XTX_reg = XTX + sigma_squared * np.eye(d_x)
                XTX_reg_inv = np.linalg.inv(XTX_reg)
                mu = XTX_reg_inv @ x_context.T @ y_context
                
                # Posterior covariance: Σ = (I + (1/σ²)X^T X)^{-1}
                Sigma = np.linalg.inv(np.eye(d_x) + (1/sigma_squared) * XTX)
                
                # Predictive mean and variance
                mean_i = x_star[0] @ mu
                var_i = sigma_squared + x_star[0] @ Sigma @ x_star[0]
            
            std_i = np.sqrt(var_i)
            
            # Compute CDF at bucket edges
            cdf_edges = stats.norm.cdf(bucket_edges, loc=mean_i, scale=std_i)
            
            # Bucket probabilities are differences in CDF
            bucket_probs = cdf_edges[1:] - cdf_edges[:-1]
            
            # Handle edge cases: add tail probabilities to edge buckets
            # Left tail (below y_min) goes to first bucket
            left_tail = cdf_edges[0]
            bucket_probs[0] += left_tail
            
            # Right tail (above y_max) goes to last bucket  
            right_tail = 1.0 - cdf_edges[-1]
            bucket_probs[-1] += right_tail
            
            # Normalize to ensure probabilities sum to 1
            bucket_probs = bucket_probs / bucket_probs.sum()
            
            discrete_probs[b, i] = bucket_probs
    
    # Convert back to torch tensor if input was tensor
    if torch.is_tensor(xs):
        discrete_probs = torch.from_numpy(discrete_probs).float().to(device)
    
    return discrete_probs


def get_model_codelength(config: ModelConfig, model: AutoregressivePFN, xs: torch.Tensor, ys: torch.Tensor): #todo: move this to utils?
    y_bins = bin_y_values(ys, config.y_min, config.y_max, config.d_vocab) # B x N
    logits = model(xs, ys) # B x 2*N x d_vocab
    logprobs = F.log_softmax(logits, dim=-1) # B x 2*N x d_vocab
    logprobs = logprobs[:, ::2, :] # B x N x d_vocab
    
    # Get negative log probs for true y values (B x N)
    neg_logprobs = -torch.gather(logprobs, -1, y_bins.unsqueeze(-1)).squeeze(-1)
    
    # Calculate running sum across sequence length (B x N)
    codelength = torch.cumsum(neg_logprobs, dim=1)

    return codelength


def get_ridge_codelength(config: ModelConfig, xs: torch.Tensor, ys: torch.Tensor, sigma_squared=0.25):
    """
    Compute Ridge regression codelength autoregressively.
    Same as get_model_codelength but uses ridge_ppd instead of model predictions.
    
    Args:
        config: Model configuration
        xs: Input points, shape (batch_size, n_context, d_x)
        ys: Output values, shape (batch_size, n_context)
        sigma_squared: Noise variance for Ridge regression
        
    Returns:
        codelength: Cumulative negative log probabilities, shape (batch_size, n_context)
    """
    # Get ridge posterior predictive distributions
    ridge_probs = ridge_ppd(xs, ys, config, sigma_squared)  # B x N x d_vocab
    ridge_logprobs = torch.log(ridge_probs + 1e-8)  # Add small epsilon for numerical stability
    
    # Bin the true y values
    y_bins = bin_y_values(ys, config.y_min, config.y_max, config.d_vocab)  # B x N
    
    # Get negative log probs for true y values (B x N)
    neg_logprobs = -torch.gather(ridge_logprobs, -1, y_bins.unsqueeze(-1)).squeeze(-1)
    
    # Calculate running sum across sequence length (B x N)
    codelength = torch.cumsum(neg_logprobs, dim=1)

    return codelength





#%%
__all__ = [
    "get_device",
    "default_model_config",
    "get_checkpoints_dir",
    "build_checkpoint_path",
    "get_pretrain_distribution_path",
    "get_true_distribution_path",
    "load_model_from_checkpoint",
    "load_task_distribution",
    "extract_w_pool",
    "ridge_ppd",
    "get_model_codelength",
    "get_ridge_codelength",
]
