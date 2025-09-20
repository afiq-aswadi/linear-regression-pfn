"""
Plot the distribution of model outputs for sampled prompts from either true or general distributions for the final checkpoint of some run.


TODO: check what happened with get_batch? might've messed up
"""
#%%

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import os
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
    build_experiment_filename,
    ensure_experiment_dir,
)
from experiments.experiment_configs import (
    RAVENTOS_SWEEP_MODEL_CONFIG,
    RUNS,
    CHECKPOINTS_DIR,
    PLOTS_DIR,
)

device = get_device()
model_config = RAVENTOS_SWEEP_MODEL_CONFIG
BASE_PLOT_DIR = ensure_experiment_dir(PLOTS_DIR, __file__)
#%%

n_batches = 16
batch_size = 16
CKPT_IDX_OVERRIDE: int | None = None  # set to an int to override per-run checkpoint selection

bin_width = (model_config.y_max - model_config.y_min) / model_config.d_vocab
bin_centers = torch.linspace(
    model_config.y_min + bin_width / 2,
    model_config.y_max - bin_width / 2,
    model_config.d_vocab,
    device=device,
    dtype=torch.float32,
)
bin_centers_np = bin_centers.detach().cpu().numpy()


#%%
# Use a single noise variance for all runs; included in plot titles NOTE: These are sampling from true distribution.
NOISE_VARIANCE = 0.25

import matplotlib.pyplot as plt
import torch.nn.functional as F

# Iterate over all configured runs and evaluate the final checkpoint (index 4)
for run_key, run_info in RUNS.items():
    run_id = run_info["run_id"]
    run_output_dir = BASE_PLOT_DIR  # Save all plots for this experiment in a single directory
   
    # Use override checkpoint if provided, otherwise default to the final checkpoint for this run
    ckpt_idx = CKPT_IDX_OVERRIDE if CKPT_IDX_OVERRIDE is not None else run_info["ckpts"][-1]

    # Build model + load checkpoint
    model_path = build_checkpoint_path(
        CHECKPOINTS_DIR, run_id, run_info["task_size"], ckpt_idx
    )
    model = load_model_from_checkpoint(model_config, model_path, device=device)

    # Load matching pretrain task distribution for this run
    # Note: "task_size" here refers to the number of tasks used during pretraining in file naming
    task_distribution_path = get_pretrain_distribution_path(
        CHECKPOINTS_DIR,
        run_id,
        run_info["task_size"],
        model_config.d_x,
    )
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

    logits = model(xs, ys)
    final_logits = logits[:, -2, :]
    probs = F.softmax(final_logits, dim=-1)
    model_mean = torch.sum(bin_centers * probs, dim=-1).detach()


    # Create a figure with subplots for each batch
    num_sequences = probs.shape[0]
    n_cols = 2
    n_rows = (num_sequences + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    # Add a figure-level title including noise_variance and task_size
    fig.suptitle(
        f"Run {run_key} ({run_id}) | ckpt={ckpt_idx} | num_tasks={run_info['task_size']} | noise_variance={NOISE_VARIANCE}",
        fontsize=12,
    )

    for i in range(num_sequences):
        prob_dist = probs[i].detach().cpu().numpy()
        axes[i].hist(
            bin_centers_np,
            weights=prob_dist,
            bins=len(bin_centers_np),
            density=True,
            color='white',
            edgecolor='black',
        )

        # Add vertical line for true y value
        true_y = final_y[i].item()
        dmmse_y = dmmse_preds[i].item()
        ridge_y = ridge_preds[i].item()
        model_mean_y = model_mean[i].item()
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
    for i in range(num_sequences, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = build_experiment_filename(
        "checkpoint-grid",
        run=run_key,
        run_id=run_id,
        tasks=run_info["task_size"],
        ckpt=ckpt_idx,
        dataset="pretrain",
        noise=NOISE_VARIANCE,
        batches=batch_size,
    )
    plot_path = os.path.join(run_output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pretrain grid: {plot_path}")

#%%

NOISE_VARIANCE = 0.25

import matplotlib.pyplot as plt
import torch.nn.functional as F

# Iterate over all configured runs and evaluate the final checkpoint (index 4)
for run_key, run_info in RUNS.items():
    run_id = run_info["run_id"]
    run_output_dir = BASE_PLOT_DIR  # Save all plots for this experiment in a single directory
    ckpt_idx = run_info["ckpts"][-1]

    # Build model + load checkpoint
    model_path = build_checkpoint_path(
        CHECKPOINTS_DIR, run_id, run_info["task_size"], ckpt_idx
    )
    model = load_model_from_checkpoint(model_config, model_path, device=device)

    # Load matching pretrain task distribution for this run
    # Note: "task_size" here refers to the number of tasks used during pretraining in file naming
    task_distribution_path = get_pretrain_distribution_path(
        CHECKPOINTS_DIR,
        run_id,
        run_info["task_size"],
        model_config.d_x,
    )
    general_task_distribution_path = get_true_distribution_path(
        CHECKPOINTS_DIR, run_id, model_config.d_x
    )  # true gaussian is same across runs

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
    probs = F.softmax(last_logit, dim=-1)
    model_mean = torch.sum(bin_centers * probs, dim=-1).detach()



    # Create a figure with subplots for each batch
    n_sequences = probs.shape[0]
    n_cols = 2
    n_rows = (n_sequences + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    # Add a figure-level title including noise_variance and task_size
    fig.suptitle(
        f"Run {run_key} ({run_id}) | ckpt={ckpt_idx} | num_tasks={run_info['task_size']} | noise_variance={NOISE_VARIANCE}",
        fontsize=12,
    )

    for i in range(n_sequences):
        prob_dist = probs[i].detach().cpu().numpy()
        axes[i].hist(
            bin_centers_np,
            weights=prob_dist,
            bins=len(bin_centers_np),
            density=True,
            color='white',
            edgecolor='black',
        )

        # Add vertical line for true y value
        true_y = final_y[i].item()
        dmmse_y = dmmse_preds[i].item()
        ridge_y = ridge_preds[i].item()
        model_mean_y = model_mean[i].item()
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
    for i in range(n_sequences, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = build_experiment_filename(
        "checkpoint-grid",
        run=run_key,
        run_id=run_id,
        tasks=run_info["task_size"],
        ckpt=ckpt_idx,
        dataset="general",
        noise=NOISE_VARIANCE,
        batches=batch_size,
    )
    plot_path = os.path.join(run_output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved general grid: {plot_path}")
# %%

print(f"\nCheckpoint grid figures saved under {BASE_PLOT_DIR} per run.")
