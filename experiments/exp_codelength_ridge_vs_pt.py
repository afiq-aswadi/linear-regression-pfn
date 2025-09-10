"""
We sample a batch of prompts and compute the codelength, comparing the ridge ppd and the model outputs.
This script creates codelength comparisons for both pretraining and generalizing distributions across all models.
"""

#%%
import os
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from models.model import AutoregressivePFN
from models.model_config import ModelConfig
from samplers.tasks import load_task_distribution_from_pt, RegressionSequenceDistribution
from baselines import dmmse_predictor, ridge_predictor

from experiments.experiment_utils import (
    get_device,
    build_checkpoint_path,
    get_pretrain_distribution_path,
    get_true_distribution_path,
    load_model_from_checkpoint,
    get_model_codelength,
    get_ridge_codelength,
    load_task_distribution
)
from experiments.experiment_configs import (
    RAVENTOS_SWEEP_MODEL_CONFIG,
    RUNS,
    CHECKPOINTS_DIR,
    PLOTS_DIR
)

def ridge_ppd(xs, ys, model_config: ModelConfig, sigma_squared=0.25):
    """
    Compute Ridge regression posterior predictive distributions autoregressively.
    For each position i, predict y_i using context from positions 0 to i-1.
    
    Args:
        xs: Input points, shape (n_context, d_x)
        ys: Output values, shape (n_context,) 
        model_config: Model configuration containing bucket parameters
        sigma_squared: Noise variance (default 0.25)
    
    Returns:
        discrete_probs: Discrete probabilities over buckets for each position, 
                       shape (n_context, d_vocab)
    """
    device = xs.device if hasattr(xs, 'device') else 'cpu'
    
    # Convert to numpy for scipy operations
    if torch.is_tensor(xs):
        xs_np = xs.detach().cpu().numpy()
        ys_np = ys.detach().cpu().numpy()
    else:
        xs_np = xs
        ys_np = ys
    
    n_context, d_x = xs_np.shape
    discrete_probs = np.zeros((n_context, model_config.d_vocab))
    
    # Setup bucket parameters
    bucket_edges = np.linspace(model_config.y_min, model_config.y_max, model_config.d_vocab + 1)
    
    for i in range(n_context):
        if i == 0:
            # First prediction: no context, use prior N(0, σ²)
            mean_i = 0.0
            var_i = sigma_squared
        else:
            # Use context from positions 0 to i-1
            x_context = xs_np[:i]  # shape (i, d_x)
            y_context = ys_np[:i]  # shape (i,)
            x_star = xs_np[i:i+1]  # shape (1, d_x)
            
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
        
        discrete_probs[i] = bucket_probs
    
    # Convert back to torch tensor if input was tensor
    if torch.is_tensor(xs):
        discrete_probs = torch.from_numpy(discrete_probs).float().to(device)
    
    return discrete_probs

#%%
device = get_device()
model_config = RAVENTOS_SWEEP_MODEL_CONFIG

# Configuration
batch_size = 8
NOISE_VARIANCE = 0.25
prompt_len = 128
task_size = 16  # Using same task_size as original

# Use final checkpoint (149999) for all models
ckpt_idx = 149999

# Generate shared datasets for consistency across models
print("Generating shared datasets for consistency across models...")

# For generalizing distribution, use the same dataset across all models (m16's true distribution)
general_dist_path = get_true_distribution_path(CHECKPOINTS_DIR, "20250826_162748", task_size)  
general_dist_distribution = load_task_distribution(general_dist_path, device=device)
general_dist_sampler = RegressionSequenceDistribution(general_dist_distribution, noise_variance=NOISE_VARIANCE)

# Generate one shared dataset for generalizing distribution
xs_general, ys_general = general_dist_sampler.get_batch(prompt_len, batch_size)
xs_general = xs_general.to(device)
ys_general = ys_general.to(device)

print(f"Generated shared generalizing dataset: xs shape {xs_general.shape}, ys shape {ys_general.shape}")

#%%
def compute_codelengths_for_model(run_key, run_info, xs, ys, distribution_type):
    """Compute codelengths for a single model on given data."""
    run_id = run_info["run_id"]
    num_tasks = run_info["task_size"]
    
    model_path = build_checkpoint_path(CHECKPOINTS_DIR, run_id, num_tasks, ckpt_idx)
    print(f"Loading model from {model_path}")
    model = load_model_from_checkpoint(model_config, model_path, device=device)
    
    codelength_model = get_model_codelength(model_config, model, xs, ys)
    codelength_ridge = get_ridge_codelength(model_config, xs, ys, sigma_squared=0.25)
    
    return codelength_model, codelength_ridge

def plot_codelengths(run_key, codelength_model, codelength_ridge, distribution_type):
    """Create and save codelength comparison plot."""
    model_cl = codelength_model.mean(dim=0).detach().cpu().numpy()
    ridge_cl = codelength_ridge.mean(dim=0).detach().cpu().numpy()
    
    plt.figure(figsize=(12, 7))
    plt.plot(model_cl, label='Model (Transformer)', linewidth=2, color='#2E86AB')
    plt.plot(ridge_cl, label='Ridge Regression', linewidth=2, color='#A23B72')
    plt.xlabel('Sequence Position')
    plt.ylabel('Cumulative Codelength (nats)')
    plt.title(f'{run_key} | Codelength Comparison | {distribution_type} Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    filename = f"codelength_{run_key}_{distribution_type}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {filename}")

#%%
# Process all models for both distributions
print("Processing all models...")

for run_key, run_info in RUNS.items():
    print(f"\nProcessing {run_key} (task_size={run_info['task_size']})...")
    
    try:
        # 1. Generalizing distribution (shared dataset)
        print(f"  Computing codelengths for generalizing distribution...")
        codelength_model_gen, codelength_ridge_gen = compute_codelengths_for_model(
            run_key, run_info, xs_general, ys_general, "generalizing"
        )
        plot_codelengths(run_key, codelength_model_gen, codelength_ridge_gen, "generalizing")
        
        # 2. Pretraining distribution (model-specific dataset)
        print(f"  Loading pretraining distribution and computing codelengths...")
        pretrain_dist_path = get_pretrain_distribution_path(CHECKPOINTS_DIR, run_info["run_id"], run_info["task_size"], task_size)
        pretrain_dist_distribution = load_task_distribution_from_pt(pretrain_dist_path, device=device)
        pretrain_dist_sampler = RegressionSequenceDistribution(pretrain_dist_distribution, noise_variance=NOISE_VARIANCE)
        
        xs_pretrain, ys_pretrain = pretrain_dist_sampler.get_batch(prompt_len, batch_size)
        xs_pretrain = xs_pretrain.to(device)
        ys_pretrain = ys_pretrain.to(device)
        
        codelength_model_pre, codelength_ridge_pre = compute_codelengths_for_model(
            run_key, run_info, xs_pretrain, ys_pretrain, "pretraining"
        )
        plot_codelengths(run_key, codelength_model_pre, codelength_ridge_pre, "pretraining")
        
    except Exception as e:
        print(f"  Error processing {run_key}: {str(e)}")
        continue

print("\nCodelength analysis complete! Check the plots/ directory for results.")

#%%


