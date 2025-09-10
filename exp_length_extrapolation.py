"""
Replication(ish) of Figure 3 of Fang et al. We sample a prompt with length greater than the length of the training prompts, and see if the model can extrapolate to this longer length.
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
    load_task_distribution
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
batch_size = 8
NOISE_VARIANCE = 0.25
prompt_len = 1024
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

def compute_mse_losses_per_position(run_key, run_info, xs, ys, dmmse_preds=None):
    """Compute MSE losses at each sequence position for a model."""
    run_id = run_info["run_id"]
    num_tasks = run_info["task_size"]
    
    model_path = build_checkpoint_path(CHECKPOINTS_DIR, run_id, num_tasks, ckpt_idx)
    model = load_model_from_checkpoint(model_config, model_path, device=device)
    
    _, avg_outputs_model = model.get_model_mean_prediction(xs, ys)
    
    # Keep ridge regression artificially constrained to model's training range for fair comparison
    try:
        avg_outputs_ridge = ridge_predictor(xs, ys, noise_variance=0.25, bound_by_y_min_max=True, y_min=model_config.y_min, y_max=model_config.y_max)
    except Exception as e:
        print(f"    Ridge regression failed: {str(e)}")
        print(f"    Using model predictions clamped to [{model_config.y_min}, {model_config.y_max}] as Ridge fallback")
        # Fallback: clamp model predictions to the training range as "ridge"
        avg_outputs_ridge = torch.clamp(avg_outputs_model, model_config.y_min, model_config.y_max).unsqueeze(-1)

    # Ensure shapes match - squeeze ys if it has extra dimension
    ys_squeezed = ys.squeeze(-1) if ys.dim() == 3 else ys
    avg_outputs_ridge_squeezed = avg_outputs_ridge.squeeze(-1) if avg_outputs_ridge.dim() == 3 else avg_outputs_ridge

    # Debug y value ranges
    if run_key == 'm1':  # Only print once
        print(f"  Y value range: min={ys_squeezed.min().item():.3f}, max={ys_squeezed.max().item():.3f}")
        print(f"  Model config y_min={model_config.y_min}, y_max={model_config.y_max}")
        print(f"  Model predictions range: min={avg_outputs_model.min().item():.3f}, max={avg_outputs_model.max().item():.3f}")
        print(f"  Ridge predictions range: min={avg_outputs_ridge_squeezed.min().item():.3f}, max={avg_outputs_ridge_squeezed.max().item():.3f}")
        
        # Count values outside model's training range
        outside_range = ((ys_squeezed < model_config.y_min) | (ys_squeezed > model_config.y_max)).sum().item()
        total_values = ys_squeezed.numel()
        percentage = (outside_range / total_values) * 100
        print(f"  Values outside model range [{model_config.y_min}, {model_config.y_max}]: {outside_range}/{total_values} ({percentage:.1f}%)")

    # Compute MSE loss at each position, averaged over batch
    mse_losses_model = []
    mse_losses_ridge = []
    # mse_losses_dmmse = []
    
    # Also collect individual MSE values for histograms
    individual_mses_model = []
    individual_mses_ridge = []
    # individual_mses_dmmse = []
    
    for pos in range(ys_squeezed.shape[1]):  # For each in-context example
        # Individual MSE values for each batch item at this position
        mse_model_batch = ((avg_outputs_model[:, pos] - ys_squeezed[:, pos]) ** 2).detach().cpu().numpy()
        mse_ridge_batch = ((avg_outputs_ridge_squeezed[:, pos] - ys_squeezed[:, pos]) ** 2).detach().cpu().numpy()
        
        individual_mses_model.extend(mse_model_batch)
        individual_mses_ridge.extend(mse_ridge_batch)
        
        # Average MSE over batch for this position
        mse_model = mse_model_batch.mean()
        mse_ridge = mse_ridge_batch.mean()
        mse_losses_model.append(mse_model)
        mse_losses_ridge.append(mse_ridge)
        
        # Add DMMSE if provided
        # if dmmse_preds is not None:
            # dmmse_squeezed = dmmse_preds.squeeze(-1) if dmmse_preds.dim() == 3 else dmmse_preds
            # mse_dmmse_batch = ((dmmse_squeezed[:, pos] - ys_squeezed[:, pos]) ** 2).detach().cpu().numpy()
            # individual_mses_dmmse.extend(mse_dmmse_batch)
            # mse_dmmse = mse_dmmse_batch.mean()
            # mse_losses_dmmse.append(mse_dmmse)
    return mse_losses_model, mse_losses_ridge, individual_mses_model, individual_mses_ridge
    # return mse_losses_model, mse_losses_ridge, mse_losses_dmmse, individual_mses_model, individual_mses_ridge, individual_mses_dmmse

# Collect data for unified plot and histograms
model_results_general = {}
model_results_pretrain = {}
ridge_results_general = {}
ridge_results_pretrain = {}
dmmse_results_general = {}
dmmse_results_pretrain = {}
model_histograms_general = {}
model_histograms_pretrain = {}
ridge_histograms_general = {}
ridge_histograms_pretrain = {}
dmmse_histograms_general = {}
dmmse_histograms_pretrain = {}

#%%
print("Processing all models...")

for run_key, run_info in RUNS.items():
    print(f"\nProcessing {run_key} (task_size={run_info['task_size']})...")
    
    try:
        # 1. Generalizing distribution (shared dataset)
        print(f"  Computing MSE losses for generalizing distribution...")
        
        # Load this model's discrete task distribution for DMMSE computation
        discrete_dist_path = get_pretrain_distribution_path(CHECKPOINTS_DIR, run_info["run_id"], run_info["task_size"], task_size)
        discrete_dist_distribution = load_task_distribution(discrete_dist_path, device=device)
        
        # Compute DMMSE predictions using this model's discrete distribution on general dataset
        # dmmse_preds_general = dmmse_predictor(xs_general, ys_general, discrete_dist_distribution, NOISE_VARIANCE, bound_by_y_min_max=True, y_min=model_config.y_min, y_max=model_config.y_max).to(device)

        mse_losses_model_gen, mse_losses_ridge_gen, individual_mses_model_gen, individual_mses_ridge_gen = compute_mse_losses_per_position(
            run_key, run_info, xs_general, ys_general,
        )

        # mse_losses_model_gen, mse_losses_ridge_gen, mse_losses_dmmse_gen, individual_mses_model_gen, individual_mses_ridge_gen, individual_mses_dmmse_gen = compute_mse_losses_per_position(
        #     run_key, run_info, xs_general, ys_general, #dmmse_preds_general
        # )
        
        model_results_general[run_key] = mse_losses_model_gen
        # dmmse_results_general[run_key] = mse_losses_dmmse_gen
        model_histograms_general[run_key] = individual_mses_model_gen
        # dmmse_histograms_general[run_key] = individual_mses_dmmse_gen
        if run_key == 'm1':  # Only need to store ridge once since it's the same for all
            ridge_results_general['Ridge'] = mse_losses_ridge_gen
            ridge_histograms_general['Ridge'] = individual_mses_ridge_gen
            
        # Create individual plot for this model - general dataset
        plt.figure(figsize=(10, 6))
        x_values = range(1, len(mse_losses_model_gen) + 1)
        plt.plot(x_values, mse_losses_model_gen, label=f'Model {run_key}', linewidth=2, color='blue', marker='o')
        plt.plot(x_values, mse_losses_ridge_gen, label='Ridge Regression', linewidth=2, color='red', linestyle='--', marker='s')
        # plt.plot(x_values, mse_losses_dmmse_gen, label='DMMSE', linewidth=2, color='green', linestyle=':', marker='^')
        plt.xlabel('Number of In-Context Examples')
        plt.ylabel('MSE Loss')
        plt.title(f'MSE Loss vs In-Context Examples - {run_key} (General Dataset)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOTS_DIR, f"individual_{run_key}_general.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Pretraining distribution (model-specific dataset)
        print(f"  Loading pretraining distribution and computing MSE losses...")
        pretrain_dist_path = get_pretrain_distribution_path(CHECKPOINTS_DIR, run_info["run_id"], run_info["task_size"], task_size)
        pretrain_dist_distribution = load_task_distribution_from_pt(pretrain_dist_path, device=device)
        pretrain_dist_sampler = RegressionSequenceDistribution(pretrain_dist_distribution, noise_variance=NOISE_VARIANCE)
        
        xs_pretrain, ys_pretrain = pretrain_dist_sampler.get_batch(prompt_len, batch_size)
        xs_pretrain = xs_pretrain.to(device)
        ys_pretrain = ys_pretrain.to(device)

        # Compute DMMSE predictions for pretraining data
        # dmmse_preds_pretrain = dmmse_predictor(xs_pretrain, ys_pretrain, pretrain_dist_distribution, NOISE_VARIANCE, bound_by_y_min_max=True, y_min=model_config.y_min, y_max=model_config.y_max).to(device)
        
        mse_losses_model_pre, mse_losses_ridge_pre, individual_mses_model_pre, individual_mses_ridge_pre = compute_mse_losses_per_position(
            run_key, run_info, xs_pretrain, ys_pretrain,
        )

        # mse_losses_model_pre, mse_losses_ridge_pre, mse_losses_dmmse_pre, individual_mses_model_pre, individual_mses_ridge_pre, individual_mses_dmmse_pre = compute_mse_losses_per_position(
        #     run_key, run_info, xs_pretrain, ys_pretrain, #dmmse_preds_pretrain
        # )
        
        model_results_pretrain[run_key] = mse_losses_model_pre
        # dmmse_results_pretrain[run_key] = mse_losses_dmmse_pre
        model_histograms_pretrain[run_key] = individual_mses_model_pre
        # dmmse_histograms_pretrain[run_key] = individual_mses_dmmse_pre
        if run_key == 'm1':  # Only need to store ridge once since it's the same for all
            ridge_results_pretrain['Ridge'] = mse_losses_ridge_pre
            ridge_histograms_pretrain['Ridge'] = individual_mses_ridge_pre
            
        # Create individual plot for this model - pretraining dataset
        plt.figure(figsize=(10, 6))
        x_values = range(1, len(mse_losses_model_pre) + 1)
        plt.plot(x_values, mse_losses_model_pre, label=f'Model {run_key}', linewidth=2, color='blue', marker='o')
        plt.plot(x_values, mse_losses_ridge_pre, label='Ridge Regression', linewidth=2, color='red', linestyle='--', marker='s')
        # plt.plot(x_values, mse_losses_dmmse_pre, label='DMMSE', linewidth=2, color='green', linestyle=':', marker='^')
        plt.xlabel('Number of In-Context Examples')
        plt.ylabel('MSE Loss')
        plt.title(f'MSE Loss vs In-Context Examples - {run_key} (Pretraining Dataset)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOTS_DIR, f"individual_{run_key}_pretrain.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"  Error processing {run_key}: {str(e)}")
        continue

# Create unified plots
print("\nCreating unified plots...")

# Unified plot - General dataset
plt.figure(figsize=(14, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(model_results_general)))
for (run_key, mse_losses), color in zip(model_results_general.items(), colors):
    task_size = RUNS[run_key]['task_size']
    x_values = range(1, len(mse_losses) + 1)
    plt.plot(x_values, mse_losses, label=f'Model {run_key} (task_size={task_size})', 
             linewidth=2, color=color, marker='o', markersize=4)

if 'Ridge' in ridge_results_general:
    x_values = range(1, len(ridge_results_general['Ridge']) + 1)
    plt.plot(x_values, ridge_results_general['Ridge'], label='Ridge Regression', 
             linewidth=2, color='black', linestyle='--', marker='s', markersize=4)

plt.xlabel('Number of In-Context Examples')
plt.ylabel('MSE Loss')
plt.title('MSE Loss vs Number of In-Context Examples (General Dataset)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR, "unified_mse_general.png"), dpi=150, bbox_inches='tight')
plt.close()

# Unified plot - Pretraining dataset
plt.figure(figsize=(14, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(model_results_pretrain)))
for (run_key, mse_losses), color in zip(model_results_pretrain.items(), colors):
    task_size = RUNS[run_key]['task_size']
    x_values = range(1, len(mse_losses) + 1)
    plt.plot(x_values, mse_losses, label=f'Model {run_key} (task_size={task_size})', 
             linewidth=2, color=color, marker='o', markersize=4)

if 'Ridge' in ridge_results_pretrain:
    x_values = range(1, len(ridge_results_pretrain['Ridge']) + 1)
    plt.plot(x_values, ridge_results_pretrain['Ridge'], label='Ridge Regression', 
             linewidth=2, color='black', linestyle='--', marker='s', markersize=4)

plt.xlabel('Number of In-Context Examples')
plt.ylabel('MSE Loss')
plt.title('MSE Loss vs Number of In-Context Examples (Pretraining Dataset)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR, "unified_mse_pretrain.png"), dpi=150, bbox_inches='tight')
plt.close()

# Create histogram plots for MSE distributions
print("\nCreating MSE distribution histograms...")

# Histogram for general dataset - all models
plt.figure(figsize=(15, 10))
n_models = len(model_histograms_general)
n_cols = 4
n_rows = (n_models + n_cols - 1) // n_cols

for i, (run_key, individual_mses) in enumerate(model_histograms_general.items()):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.hist(individual_mses, bins=50, alpha=0.7, density=True, label=f'Model {run_key}')
    if 'Ridge' in ridge_histograms_general:
        plt.hist(ridge_histograms_general['Ridge'], bins=50, alpha=0.7, density=True, label='Ridge', color='red')
    plt.xlabel('MSE')
    plt.ylabel('Density')
    plt.title(f'{run_key} (task_size={RUNS[run_key]["task_size"]})')
    plt.legend()
    plt.yscale('linear')
    plt.grid(True, alpha=0.3)

plt.suptitle('MSE Distribution Histograms - General Dataset')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "mse_histograms_general.png"), dpi=150, bbox_inches='tight')
plt.close()

# Individual histogram plots for better readability - General Dataset
# for run_key, individual_mses in model_histograms_general.items():
#     plt.figure(figsize=(10, 6))
#     plt.hist(individual_mses, bins=100, alpha=0.7, density=True, label=f'Model {run_key}', color='blue')
#     if 'Ridge' in ridge_histograms_general:
#         plt.hist(ridge_histograms_general['Ridge'], bins=100, alpha=0.7, density=True, label='Ridge', color='red')
#     # if run_key in dmmse_histograms_general:
#         # plt.hist(dmmse_histograms_general[run_key], bins=100, alpha=0.7, density=True, label='DMMSE', color='green')
#     plt.xlabel('MSE')
#     plt.ylabel('Density')
#     plt.title(f'MSE Distribution - {run_key} vs Baselines (General Dataset)')
#     plt.legend()
#     plt.yscale('linear')
#     plt.grid(True, alpha=0.3)
#     plt.savefig(os.path.join(PLOTS_DIR, f"histogram_{run_key}_general.png"), dpi=150, bbox_inches='tight')
#     plt.close()

# Add histogram plots for pretrain data
print("Creating MSE distribution histograms for pretrain data...")

# # Histogram for pretrain dataset - all models
# plt.figure(figsize=(15, 10))
# n_models = len(model_histograms_pretrain)
# n_cols = 4
# n_rows = (n_models + n_cols - 1) // n_cols

# for i, (run_key, individual_mses) in enumerate(model_histograms_pretrain.items()):
#     plt.subplot(n_rows, n_cols, i + 1)
#     plt.hist(individual_mses, bins=50, alpha=0.7, density=True, label=f'Model {run_key}')
#     if 'Ridge' in ridge_histograms_pretrain:
#         plt.hist(ridge_histograms_pretrain['Ridge'], bins=50, alpha=0.7, density=True, label='Ridge', color='red')
#     plt.xlabel('MSE')
#     plt.ylabel('Density')
#     plt.title(f'{run_key} (task_size={RUNS[run_key]["task_size"]})')
#     plt.legend()
#     plt.yscale('linear')
#     plt.grid(True, alpha=0.3)

# plt.suptitle('MSE Distribution Histograms - Pretrain Dataset')
# plt.tight_layout()
# plt.savefig(os.path.join(PLOTS_DIR, "mse_histograms_pretrain.png"), dpi=150, bbox_inches='tight')
# plt.close()

# # Individual histogram plots for pretrain data
# for run_key, individual_mses in model_histograms_pretrain.items():
#     plt.figure(figsize=(10, 6))
#     plt.hist(individual_mses, bins=100, alpha=0.7, density=True, label=f'Model {run_key}', color='blue')
#     if 'Ridge' in ridge_histograms_pretrain:
#         plt.hist(ridge_histograms_pretrain['Ridge'], bins=100, alpha=0.7, density=True, label='Ridge', color='red')
#     if run_key in dmmse_histograms_pretrain:
#         plt.hist(dmmse_histograms_pretrain[run_key], bins=100, alpha=0.7, density=True, label='DMMSE', color='green')
#     plt.xlabel('MSE')
#     plt.ylabel('Density')
#     plt.title(f'MSE Distribution - {run_key} vs Baselines (Pretrain Dataset)')
#     plt.legend()
#     plt.yscale('linear')
#     plt.grid(True, alpha=0.3)
#     plt.savefig(os.path.join(PLOTS_DIR, f"histogram_{run_key}_pretrain.png"), dpi=150, bbox_inches='tight')
#     plt.close()

print(f"Saved unified general dataset plot: unified_mse_general.png")
print(f"Saved unified pretraining dataset plot: unified_mse_pretrain.png")
print(f"Saved individual plots for each model and dataset combination")
print(f"Saved MSE histogram plots: mse_histograms_general.png and individual histogram_[model]_general.png")
print(f"Saved MSE histogram plots: mse_histograms_pretrain.png and individual histogram_[model]_pretrain.png")

print("\nMSE losses analysis complete! Check the plots/ directory for results.")
# %%
