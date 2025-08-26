#%%

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from models.model import AutoregressivePFN
from models.model_config import ModelConfig
from predictive_resampling.predictive_resampling import predictive_resampling_beta_chunked
from samplers.tasks import load_task_distribution_from_pt

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


training_config = {
    'device': 'cuda',
    'task_size': 2,
    'num_tasks': 2,
    'noise_var': .25,
    'num_examples': 64,
    'learning_rate': 0.003,
    'training_steps': 100000,
    'batch_size': 256,
    'eval_batch_size': 1024,
    'print_loss_interval': 100,
    'print_metrics_interval': 1000,
    'n_checkpoints': 10,
}

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
    "m7": {"run_id": "20250826_011959", "task_size": 2, "ckpts": [0, 1, 2, 3, 4]},
}


# Iterate over runs and plot distributions across checkpoints
for run_key, run_info in RUNS.items():
    run_id = run_info["run_id"]
    ckpt_indices = run_info["ckpts"]

    num_cols = len(ckpt_indices)
    fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 6), sharey='row')

    task_distribution_path = os.path.join(
        CHECKPOINTS_DIR,
        f"{run_id}_pretrain_discrete_{run_info['task_size']}tasks_2d.pt",
    )
    task_distribution = load_task_distribution_from_pt(task_distribution_path, device=device)
    # Extract w_pool (M x D) from loaded distribution (supports dict/object/tensor)
    w_pool = None
    if isinstance(task_distribution, dict) and 'tasks' in task_distribution:
        w_pool = task_distribution['tasks']
    elif hasattr(task_distribution, 'tasks'):
        w_pool = getattr(task_distribution, 'tasks')
    else:
        # Fallback: assume the returned object is directly the tasks tensor/ndarray
        w_pool = task_distribution
    if isinstance(w_pool, torch.Tensor):
        w_pool = w_pool.detach().cpu().numpy()
    M = int(w_pool.shape[0]) if hasattr(w_pool, 'shape') else 0

    # Ensure axes is a 2D array for consistent indexing
    if num_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col_idx, ckpt_idx in enumerate(ckpt_indices):
        model = AutoregressivePFN(model_config).to(device)
        model_path = os.path.join(
            CHECKPOINTS_DIR, f"{run_id}_model_checkpoint_{ckpt_idx}.pt"
        )
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)

        beta_hat = predictive_resampling_beta_chunked(
            model,
            model_config,
            forward_recursion_steps=64,
            forward_recursion_samples=10000,
            sample_y=True,
            save_y=False,
            init_x=None,
            init_y=None,
        )

        # Row 0: β1
        ax_top = axes[0, col_idx]
        counts1, bins1, _ = ax_top.hist(beta_hat[:, 0], bins=50, alpha=0.7, density=True)
        x1 = np.linspace(min(bins1), max(bins1), 200)
        ax_top.plot(x1, norm.pdf(x1, 0, 1), 'r-', lw=2, label='N(0,1)')
        # Overlay pretraining tasks for β1
        if M > 0:
            w_vals_0 = w_pool[:, 0]
            if M <= 20:
                ymin, ymax = ax_top.get_ylim()
                y_offset = ymin + 0.02 * (ymax - ymin)
                ax_top.plot(
                    w_vals_0,
                    [y_offset] * len(w_vals_0),
                    marker='|', linestyle='', color='darkorange',
                    markersize=20, markeredgewidth=2,
                    label='pretraining tasks' if col_idx == 0 else None,
                )
            else:
                ax_top.hist(
                    w_vals_0, bins=min(30, M), density=True,
                    color='darkorange', alpha=0.3, edgecolor='none',
                    label='pretraining tasks' if col_idx == 0 else None,
                )
        if col_idx == 0:
            ax_top.set_ylabel('β₁ density')
        ax_top.set_title(f'ckpt {ckpt_idx}')
        ax_top.grid(True, alpha=0.3)

        # Row 1: β2
        ax_bottom = axes[1, col_idx]
        counts2, bins2, _ = ax_bottom.hist(beta_hat[:, 1], bins=50, alpha=0.7, density=True)
        x2 = np.linspace(min(bins2), max(bins2), 200)
        ax_bottom.plot(x2, norm.pdf(x2, 0, 1), 'r-', lw=2, label='N(0,1)')
        # Overlay pretraining tasks for β2
        if M > 0:
            w_vals_1 = w_pool[:, 1]
            if M <= 20:
                ymin, ymax = ax_bottom.get_ylim()
                y_offset = ymin + 0.02 * (ymax - ymin)
                ax_bottom.plot(
                    w_vals_1,
                    [y_offset] * len(w_vals_1),
                    marker='|', linestyle='', color='darkorange',
                    markersize=20, markeredgewidth=2,
                    label='pretraining tasks' if col_idx == 0 else None,
                )
            else:
                ax_bottom.hist(
                    w_vals_1, bins=min(30, M), density=True,
                    color='darkorange', alpha=0.3, edgecolor='none',
                    label='pretraining tasks' if col_idx == 0 else None,
                )
        if col_idx == 0:
            ax_bottom.set_ylabel('β₂ density')
        ax_bottom.set_xlabel('β value')
        ax_bottom.grid(True, alpha=0.3)

    fig.suptitle(f"Run {run_key} ({run_id}) — task_size={run_info.get('task_size')}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
# %%
