"""
This script performs predictive resampling without conditioning on any input data across different models and checkpoints.
"""

#%%
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from models.model import AutoregressivePFN
from models.config import ModelConfig
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
    extract_w_pool,
    build_experiment_filename,
    ensure_experiment_dir,
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
BASE_PLOT_DIR = ensure_experiment_dir(PLOTS_DIR, __file__)

# Optional hard-coded overrides for quick notebook tinkering
DEFAULT_PLOT_DIMS = None  # e.g. set to 4 to always plot 4 dimensions
DEFAULT_RUN_KEYS = None   # e.g. set to ["m1", "m5"] to focus on specific runs

# Allow overriding the number of plotted dimensions via CLI or env var
plot_arg_parser = argparse.ArgumentParser(add_help=False)
plot_arg_parser.add_argument(
    "--plot-dims",
    type=int,
    default=None,
    help="Number of output dimensions to plot per checkpoint.",
)
plot_arg_parser.add_argument(
    "--run-keys",
    nargs="+",
    default=None,
    help="Subset of run keys from RUNS to process (e.g. m1 m3 m5).",
)
_plot_args, _remaining_argv = plot_arg_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

if _plot_args.plot_dims is not None and _plot_args.plot_dims < 1:
    raise ValueError("--plot-dims must be a positive integer.")

plot_dims_override = _plot_args.plot_dims
if plot_dims_override is None:
    env_override = os.getenv("PLOT_DIMS")
    if env_override is not None:
        try:
            plot_dims_override = int(env_override)
        except ValueError as exc:
            raise ValueError("Environment variable PLOT_DIMS must be an integer.") from exc
        if plot_dims_override < 1:
            raise ValueError("Environment variable PLOT_DIMS must be >= 1.")
if plot_dims_override is None and DEFAULT_PLOT_DIMS is not None:
    if DEFAULT_PLOT_DIMS < 1:
        raise ValueError("DEFAULT_PLOT_DIMS must be >= 1 when set.")
    plot_dims_override = DEFAULT_PLOT_DIMS

run_keys_override = _plot_args.run_keys
if run_keys_override is None:
    env_run_keys = os.getenv("RUN_KEYS")
    if env_run_keys:
        run_keys_override = [key.strip() for key in env_run_keys.split(",") if key.strip()]
if run_keys_override is None and DEFAULT_RUN_KEYS is not None:
    run_keys_override = list(DEFAULT_RUN_KEYS)

if run_keys_override is not None:
    missing = [key for key in run_keys_override if key not in RUNS]
    if missing:
        raise ValueError(f"Unknown run keys requested: {missing}. Valid keys: {sorted(RUNS)}")
    runs_iterable = [(key, RUNS[key]) for key in run_keys_override]
else:
    runs_iterable = list(RUNS.items())

#%%
# Configuration
forward_recursion_steps = 64
forward_recursion_samples = 1000

# Iterate through models and checkpoints
for run_key, run_info in runs_iterable:
    print(f"\nProcessing {run_key} (task_size={run_info['task_size']})...")
    run_output_dir = BASE_PLOT_DIR  # Save all plots for this experiment in a single directory
    
    # Get every 4th checkpoint
    selected_checkpoints = run_info['ckpts'][::4]
    print(f"Selected checkpoints: {selected_checkpoints}")
    
    # Determine how many dimensions the model actually supports (default to config d_x)
    full_task_dims = getattr(model_config, "d_x", None) or run_info["task_size"]
    n_dims = full_task_dims if plot_dims_override is None else min(plot_dims_override, full_task_dims)
    if plot_dims_override is not None and plot_dims_override > full_task_dims:
        print(
            f"  Requested {plot_dims_override} dims to plot but only {full_task_dims} available; plotting {full_task_dims}."
        )

    # Create figure for this model: n_dims rows (dimensions) x len(selected_checkpoints) columns
    n_checkpoints = len(selected_checkpoints)

    fig, axes = plt.subplots(n_dims, n_checkpoints, figsize=(4*n_checkpoints, 3*n_dims))
    if n_checkpoints == 1:
        axes = axes.reshape(-1, 1)
    if n_dims == 1:
        axes = axes.reshape(1, -1)
    
    # Load task distribution for this model
    try:
        pretrain_task_dims = getattr(model_config, "d_x", run_info["task_size"])
        pretrain_dist_path = get_pretrain_distribution_path(
            CHECKPOINTS_DIR,
            run_info["run_id"],
            run_info["task_size"],
            task_size=pretrain_task_dims,
        )
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
            
            # Plot the requested dimensions
            available_dims = min(n_dims, beta_hat.shape[1])
            for dim in range(available_dims):
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
    plot_filename = build_experiment_filename(
        "predictive-resampling",
        run=run_key,
        tasks=run_info["task_size"],
        ckpt_min=min(selected_checkpoints),
        ckpt_max=max(selected_checkpoints),
        ckpt_count=len(selected_checkpoints),
        recursion_steps=forward_recursion_steps,
        samples=forward_recursion_samples,
    )
    plot_path = os.path.join(run_output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    
    plt.close()  # Close to free memory

print(f"\nPredictive resampling analysis complete! Check {BASE_PLOT_DIR} for outputs.")

# %%
