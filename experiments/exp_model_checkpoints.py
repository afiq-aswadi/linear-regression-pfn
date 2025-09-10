"""
Plot the distribution of model outputs for sampled prompts from either true or general distributions for the final checkpoint of some run.


TODO: check what happened with get_batch? might've messed up
"""
#%%

import torch

from samplers.tasks import RegressionSequenceDistribution
from baselines import dmmse_predictor, ridge_predictor
from experiments.experiment_utils import (
    get_device,
    build_checkpoint_path,
    get_pretrain_distribution_path,
    get_true_distribution_path,
    load_model_from_checkpoint,
    load_task_distribution,
)
from experiments.experiment_configs import (
    RAVENTOS_SWEEP_MODEL_CONFIG,
    RUNS,
    CHECKPOINTS_DIR,
)

device = get_device()
model_config = RAVENTOS_SWEEP_MODEL_CONFIG
#%%

n_batches = 16
batch_size = 16
indices = torch.linspace(model_config.y_min, model_config.y_max, model_config.n_bins)
ckpt_idx = 149999  #what checkpoint we want to use


#%%
# Use a single noise variance for all runs; included in plot titles NOTE: These are sampling from true distribution.
NOISE_VARIANCE = 0.25

import matplotlib.pyplot as plt
import torch.nn.functional as F

# Iterate over all configured runs and evaluate the final checkpoint (index 4)
for run_key, run_info in RUNS.items():
    run_id = run_info["run_id"]
   
    # Build model + load checkpoint
    model_path = build_checkpoint_path(CHECKPOINTS_DIR, run_id, ckpt_idx)
    model = load_model_from_checkpoint(model_config, model_path, device=device)

    # Load matching pretrain task distribution for this run
    # Note: "task_size" here refers to the number of tasks used during pretraining in file naming
    task_distribution_path = get_pretrain_distribution_path(CHECKPOINTS_DIR, run_id, run_info['task_size'])
    task_distribution = load_task_distribution(task_distribution_path, device=device)

    regression_sequence_distribution = RegressionSequenceDistribution(
        task_distribution, noise_variance=NOISE_VARIANCE
    ).to(device)

    xs, ys = regression_sequence_distribution.get_batch(n_batches, batch_size)
    xs = xs.to(device)  # (batch_size, seq_len, d_x)
    ys = ys.to(device)  # (batch_size, seq_len, d_y)

    final_y = ys[:, -1, :]
    dmmse_preds = dmmse_predictor(xs, ys, task_distribution, NOISE_VARIANCE)[:,-1,:].cpu()
    ridge_preds = ridge_predictor(xs, ys, NOISE_VARIANCE)[:,-1,:].cpu()

    probs, model_mean = model.get_model_mean_prediction(xs, ys)


    # Create a figure with subplots for each batch
    n_cols = 2
    n_rows = (n_batches + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    # Add a figure-level title including noise_variance and task_size
    fig.suptitle(
        f"Run {run_key} ({run_id}) | ckpt={ckpt_idx} | num_tasks={run_info['task_size']} | noise_variance={NOISE_VARIANCE}",
        fontsize=12,
    )

    for i in range(n_batches):
        prob_dist = probs[i].detach().cpu().numpy()
        axes[i].hist(indices, weights=prob_dist, bins=len(indices), density=True, color='white', edgecolor='black')

        # Add vertical line for true y value
        true_y = final_y[i].item()
        dmmse_y = dmmse_preds[i].item()
        ridge_y = ridge_preds[i].item()
        model_mean_y = model_mean[i]
        axes[i].axvline(x=true_y, color='red', linestyle='--', label='True y', linewidth=2)
        axes[i].axvline(x=dmmse_y, color='lime', linestyle='--', label='DMMSE', linewidth=2)
        axes[i].axvline(x=ridge_y, color='blue', linestyle='--', label='Ridge', linewidth=2)
        axes[i].axvline(x=model_mean_y, color='green', linestyle='--', label='Model Mean', linewidth=2)
        axes[i].set_title(f'Batch {i} Probability Distribution\nTrue y: {true_y:.3f}')
        axes[i].set_xlabel('y value')
        axes[i].set_ylabel('Probability')
        axes[i].grid(True)
        axes[i].legend()

    # Hide any empty subplots
    for i in range(n_batches, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

#%%

NOISE_VARIANCE = 0.25

import matplotlib.pyplot as plt
import torch.nn.functional as F

# Iterate over all configured runs and evaluate the final checkpoint (index 4)
for run_key, run_info in RUNS.items():
    run_id = run_info["run_id"]
    ckpt_idx = 2  # explicitly use the final checkpoint

    # Build model + load checkpoint
    model_path = build_checkpoint_path(CHECKPOINTS_DIR, run_id, ckpt_idx)
    model = load_model_from_checkpoint(model_config, model_path, device=device)

    # Load matching pretrain task distribution for this run
    # Note: "task_size" here refers to the number of tasks used during pretraining in file naming
    task_distribution_path = get_pretrain_distribution_path(CHECKPOINTS_DIR, run_id, run_info['task_size'])
    general_task_distribution_path = get_true_distribution_path(CHECKPOINTS_DIR, run_id) # true gaussian is same across runs

    general_task_distribution = load_task_distribution(general_task_distribution_path, device=device)
    task_distribution = load_task_distribution(task_distribution_path, device=device)

    regression_sequence_distribution = RegressionSequenceDistribution(
        general_task_distribution, noise_variance=NOISE_VARIANCE
    ).to(device)

    xs, ys = regression_sequence_distribution.get_batch(n_batches, batch_size)
    xs = xs.to(device)  # (batch_size, seq_len, d_x)
    ys = ys.to(device)  # (batch_size, seq_len, d_y)

    final_y = ys[:, -1, :]
    dmmse_preds = dmmse_predictor(xs, ys, task_distribution, NOISE_VARIANCE)[:,-1,:].cpu()
    ridge_preds = ridge_predictor(xs, ys, NOISE_VARIANCE)[:,-1,:].cpu()

    logits = model(xs, ys)

    last_logit = logits[:, -2, :]

    # Convert logits to probabilities using softmax
    probs = F.softmax(last_logit, dim=-1).to('cpu')
    model_mean = torch.sum(indices * probs, dim=-1).detach()



    # Create a figure with subplots for each batch
    n_batches = probs.shape[0]
    n_cols = 2
    n_rows = (n_batches + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    # Add a figure-level title including noise_variance and task_size
    fig.suptitle(
        f"Run {run_key} ({run_id}) | ckpt={ckpt_idx} | num_tasks={run_info['task_size']} | noise_variance={NOISE_VARIANCE}",
        fontsize=12,
    )

    for i in range(n_batches):
        prob_dist = probs[i].detach().cpu().numpy()
        axes[i].hist(indices, weights=prob_dist, bins=len(indices), density=True, color='white', edgecolor='black')

        # Add vertical line for true y value
        true_y = final_y[i].item()
        dmmse_y = dmmse_preds[i].item()
        ridge_y = ridge_preds[i].item()
        model_mean_y = model_mean[i]
        axes[i].axvline(x=true_y, color='red', linestyle='--', label='True y', linewidth=2)
        axes[i].axvline(x=dmmse_y, color='lime', linestyle='--', label='DMMSE', linewidth=2)
        axes[i].axvline(x=ridge_y, color='blue', linestyle='--', label='Ridge', linewidth=2)
        axes[i].axvline(x=model_mean_y, color='green', linestyle='--', label='Model Mean', linewidth=2)
        axes[i].set_title(f'Batch {i} Probability Distribution\nTrue y: {true_y:.3f}')
        axes[i].set_xlabel('y value')
        axes[i].set_ylabel('Probability')
        axes[i].grid(True)
        axes[i].legend()

    # Hide any empty subplots
    for i in range(n_batches, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
# %%
