"""
Some cheap experiments to make sure models are trained properly.
"""
#%%

import os
import torch

from models.model import AutoregressivePFN
from models.model_config import ModelConfig
from samplers.tasks import load_task_distribution_from_pt, RegressionSequenceDistribution
from baselines import dmmse_predictor, ridge_predictor

device = "cuda" if torch.cuda.is_available() else "cpu"
#%%

model_config = ModelConfig(
    d_model=64,
    d_x=2,
    d_y=1,
    n_layers=2,
    n_heads=2,
    d_mlp=4 * 64,
    d_vocab=64,
    n_ctx=128
)

y_min = -6.0
y_max = 6.0
n_bins = 64

indices = torch.linspace(y_min, y_max, n_bins)

#%%
BASE_DIR = os.path.dirname(__file__)
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

#%%
RUNS = {
    # "m1": {"run_id": "20250818_143023", "task_size": 1, "ckpts": [0, 1, 2, 3, 4]},
    # "m2": {"run_id": "20250818_170107", "task_size": 16, "ckpts": [0, 1, 2, 3, 4]},
    # "m3": {"run_id": "20250818_194416", "task_size": 256, "ckpts": [0, 1, 2, 3, 4]},
    # "m4": {"run_id": "20250818_222551", "task_size": 4096, "ckpts": [0, 1, 2, 3, 4]},
    # "m5": {"run_id": "20250819_010712", "task_size": 65536, "ckpts": [0, 1, 2, 3, 4]},
    # "m6": {"run_id": "20250819_034904", "task_size": 1048576, "ckpts": [0, 1, 2, 3, 4]},
    "m7": {"run_id": "20250826_020546", "task_size": 2, "ckpts": [0, 1, 2, 3, 4]},
}

#%%
# Use a single noise variance for all runs; included in plot titles NOTE: These are sampling from true distribution.
NOISE_VARIANCE = 0.25

import matplotlib.pyplot as plt
import torch.nn.functional as F

# Iterate over all configured runs and evaluate the final checkpoint (index 4)
for run_key, run_info in RUNS.items():
    run_id = run_info["run_id"]
    final_ckpt_idx = 1  # explicitly use the final checkpoint

    # Build model + load checkpoint
    model = AutoregressivePFN(model_config).to(device)
    model_path = os.path.join(
        CHECKPOINTS_DIR, f"{run_id}_model_checkpoint_{final_ckpt_idx}.pt"
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    # Load matching pretrain task distribution for this run
    # Note: "task_size" here refers to the number of tasks used during pretraining in file naming
    task_distribution_path = os.path.join(
        CHECKPOINTS_DIR,
        f"{run_id}_pretrain_discrete_{run_info['task_size']}tasks_2d.pt",
    )
    task_distribution = load_task_distribution_from_pt(task_distribution_path, device=device)

    regression_sequence_distribution = RegressionSequenceDistribution(
        task_distribution, noise_variance=NOISE_VARIANCE
    ).to(device)

    xs, ys = regression_sequence_distribution.get_batch(16, 16)
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
        f"Run {run_key} ({run_id}) | ckpt={final_ckpt_idx} | num_tasks={run_info['task_size']} | noise_variance={NOISE_VARIANCE}",
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
print(dmmse_preds.shape)
print(ridge_preds.shape)

#%%

NOISE_VARIANCE = 0.25

import matplotlib.pyplot as plt
import torch.nn.functional as F

# Iterate over all configured runs and evaluate the final checkpoint (index 4)
for run_key, run_info in RUNS.items():
    run_id = run_info["run_id"]
    final_ckpt_idx = 2  # explicitly use the final checkpoint

    # Build model + load checkpoint
    model = AutoregressivePFN(model_config).to(device)
    model_path = os.path.join(
        CHECKPOINTS_DIR, f"{run_id}_model_checkpoint_{final_ckpt_idx}.pt"
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    # Load matching pretrain task distribution for this run
    # Note: "task_size" here refers to the number of tasks used during pretraining in file naming
    task_distribution_path = os.path.join(
        CHECKPOINTS_DIR,
        f"{run_id}_pretrain_discrete_{run_info['task_size']}tasks_2d.pt",
    )

    general_task_distribution_path = os.path.join(
        CHECKPOINTS_DIR,
        f"{run_id}_true_gaussian_2d.pt",
    ) #ermm this is really inefficient, the true gaussian is the same for all runs. 

    general_task_distribution = load_task_distribution_from_pt(general_task_distribution_path, device=device)
    task_distribution = load_task_distribution_from_pt(task_distribution_path, device=device)

    regression_sequence_distribution = RegressionSequenceDistribution(
        general_task_distribution, noise_variance=NOISE_VARIANCE
    ).to(device)

    xs, ys = regression_sequence_distribution.get_batch(16, 16)
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
        f"Run {run_key} ({run_id}) | ckpt={final_ckpt_idx} | num_tasks={run_info['task_size']} | noise_variance={NOISE_VARIANCE}",
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
