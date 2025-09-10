"""
This script performs predictive resampling without conditioning on any input data across different models and checkpoints.
"""

#%%
import os
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from models.model import AutoregressivePFN
from models.model_config import ModelConfig
from samplers.tasks import load_task_distribution_from_pt, RegressionSequenceDistribution
from predictive_resampling.predictive_resampling import predictive_resampling_beta_chunked, predictive_resampling_beta
from baselines import dmmse_predictor, ridge_predictor

from experiments.experiment_utils import (
    get_device,
    build_checkpoint_path,
    get_pretrain_distribution_path,
    get_true_distribution_path,
    load_model_from_checkpoint,
    get_model_codelength,
    get_ridge_codelength,
    load_task_distribution,
    extract_w_pool
)
from experiments.experiment_configs import (
    RAVENTOS_SWEEP_MODEL_CONFIG,
    RUNS,
    CHECKPOINTS_DIR,
    PLOTS_DIR
)


#%%
device = get_device()
model_config = RAVENTOS_SWEEP_MODEL_CONFIG

# Configuration
forward_recursion_steps = 64
forward_recursion_samples = 1000

# Iterate through models and checkpoints
for run_key, run_info in RUNS.items():
    print(f"\nProcessing {run_key} (task_size={run_info['task_size']})...")
    
    # Get every 4th checkpoint
    selected_checkpoints = run_info['ckpts'][::4]
    print(f"Selected checkpoints: {selected_checkpoints}")
    
    # Create figure for this model: 16 rows (dimensions) x len(selected_checkpoints) columns
    n_dims = 16  # All dimensions
    n_checkpoints = len(selected_checkpoints)
    
    fig, axes = plt.subplots(n_dims, n_checkpoints, figsize=(4*n_checkpoints, 3*n_dims))
    if n_checkpoints == 1:
        axes = axes.reshape(-1, 1)
    if n_dims == 1:
        axes = axes.reshape(1, -1)
    
    # Load task distribution for this model
    try:
        pretrain_dist_path = get_pretrain_distribution_path(CHECKPOINTS_DIR, run_info["run_id"], run_info["task_size"], task_size = n_dims)
        task_distribution = load_task_distribution(pretrain_dist_path, device=device)
        print(f"Loaded task distribution from {pretrain_dist_path}")
    except Exception as e:
        print(f"Warning: Could not load task distribution: {e}")
        task_distribution = None
 
    # Process each checkpoint
    for ckpt_idx, checkpoint_step in enumerate(selected_checkpoints):
        print(f"  Processing checkpoint {checkpoint_step}...")
        
        try:
            # Build checkpoint path and load model
            checkpoint_path = build_checkpoint_path(CHECKPOINTS_DIR, run_info["run_id"], run_info["task_size"], checkpoint_step)
            model = load_model_from_checkpoint(model_config, checkpoint_path, device=device)
            
            # Perform predictive resampling
            beta_hat = predictive_resampling_beta_chunked(
                model, model_config, 
                forward_recursion_steps=forward_recursion_steps, 
                forward_recursion_samples=forward_recursion_samples
            )
            
            # Plot all 16 dimensions
            for dim in range(n_dims):
                ax = axes[dim, ckpt_idx]
                
                # Get beta values for this dimension
                beta_values = beta_hat[:, dim]
                
                # Plot histogram
                counts, bins, _ = ax.hist(beta_values, bins=50, alpha=0.5, density=True, 
                                         color='blue', label='Beta histogram')
                
                # Plot KDE
                if len(beta_values) > 1:
                    kde = gaussian_kde(beta_values)
                    x_range = np.linspace(beta_values.min(), beta_values.max(), 200)
                    kde_values = kde(x_range)
                    ax.plot(x_range, kde_values, 'g-', lw=2, label='Beta KDE')
                
                # Plot N(0,1)
                x_normal = np.linspace(-4, 4, 200)
                ax.plot(x_normal, stats.norm.pdf(x_normal, 0, 1), 'r-', lw=2, label='N(0,1)')
                
                # Plot task distribution if available
                if task_distribution is not None:
                    task_betas = extract_w_pool(task_distribution)[:, dim]
                    
                    if len(task_betas) > 256:  # Use histogram for large number of tasks
                        ax.hist(task_betas, bins=30, alpha=0.3, density=True, 
                               color='orange', label='Task dist (hist)')
                    else:  # Use lines for small number of tasks
                        # Create a simple line plot showing the discrete values
                        unique_vals, counts = np.unique(task_betas, return_counts=True)
                        # Normalize counts to make it comparable with density plots
                        normalized_counts = counts / (len(task_betas) * (unique_vals[1] - unique_vals[0]) if len(unique_vals) > 1 else 1)
                        ax.scatter(unique_vals, normalized_counts, color='orange', s=30, label='Task dist (points)', alpha=0.7)
                        for i, (val, count) in enumerate(zip(unique_vals, normalized_counts)):
                            ax.vlines(val, 0, count, colors='orange', alpha=0.5, linewidth=1)
                
                # Formatting
                ax.set_xlabel(f'β_{dim+1}')
                ax.set_ylabel('Density')
                ax.set_title(f'Dim {dim+1}, Step {checkpoint_step}')
                ax.grid(True, alpha=0.3)
                
                # Add legend only to the first subplot to avoid clutter
                if dim == 0 and ckpt_idx == 0:
                    ax.legend(fontsize='small')
            
        except Exception as e:
            print(f"    Error processing checkpoint {checkpoint_step}: {e}")
            # Fill with empty plots if error occurs
            for dim in range(n_dims):
                ax = axes[dim, ckpt_idx]
                ax.text(0.5, 0.5, f'Error\nStep {checkpoint_step}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Dim {dim+1}, Step {checkpoint_step} (Error)')
    
    # Overall title and layout
    fig.suptitle(f'Predictive Resampling Analysis - {run_key} (task_size={run_info["task_size"]})', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'predictive_resampling_{run_key}_task_size_{run_info["task_size"]}.png'
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    
    plt.close()  # Close to free memory

print("\nPredictive resampling analysis complete!")

# %%
