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

y_min = -3.0
y_max = 3.0
n_bins = 64

indices = torch.linspace(y_min, y_max, n_bins)

#%%
BASE_DIR = os.path.dirname(__file__)
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

#%%
RUNS = {
    # "m1": {"run_id": "20250818_143023", "task_size": 1, "ckpts": [0, 1, 2, 3, 4]},
    # "m2": {"run_id": "20250818_170107", "task_size": 16, "ckpts": [0, 1, 2, 3, 4]},
     "m3": {"run_id": "20250818_194416", "task_size": 256, "ckpts": [4]},
    # "m4": {"run_id": "20250818_222551", "task_size": 4096, "ckpts": [0, 1, 2, 3, 4]},
    # "m5": {"run_id": "20250819_010712", "task_size": 65536, "ckpts": [0, 1, 2, 3, 4]},
    # "m6": {"run_id": "20250819_034904", "task_size": 1048576, "ckpts": [0, 1, 2, 3, 4]},
    # "m7": {"run_id": "20250826_020546", "task_size": 2, "ckpts": [4]},
}

batch_size = 4

#%%
# Use a single noise variance for all runs; included in plot titles NOTE: These are sampling from true distribution.
NOISE_VARIANCE = 0

import matplotlib.pyplot as plt
import torch.nn.functional as F


def _load_distribution_paths(run_id: str, task_size: int):
    pretrain_path = os.path.join(
        CHECKPOINTS_DIR,
        f"{run_id}_pretrain_discrete_{task_size}tasks_2d.pt",
    )
    general_path = os.path.join(
        CHECKPOINTS_DIR,
        f"{run_id}_true_gaussian_2d.pt",
    )
    return pretrain_path, general_path


def _compute_model_outputs_for_ckpt(run_id: str, ckpt_idx: int, xs: torch.Tensor, ys: torch.Tensor):
    model = AutoregressivePFN(model_config).to(device)
    model_path = os.path.join(CHECKPOINTS_DIR, f"{run_id}_model_checkpoint_{ckpt_idx}.pt")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    logits = model(xs, ys)
    last_logit = logits[:, -2, :]
    probs = F.softmax(last_logit, dim=-1).to("cpu")
    model_mean = torch.sum(indices * probs, dim=-1).detach()
    return probs, model_mean


def _plot_grid_across_checkpoints(
    run_key: str,
    run_id: str,
    ckpt_indices: list,
    xs: torch.Tensor,
    ys: torch.Tensor,
    dmmse_preds: torch.Tensor,
    ridge_preds: torch.Tensor,
    probs_per_ckpt: list,
    model_means_per_ckpt: list,
    title_suffix: str,
):
    final_y = ys[:, -1, :]
    n_batches = xs.shape[0]
    n_ckpts = len(ckpt_indices)

    fig, axes = plt.subplots(
        n_batches,
        n_ckpts,
        figsize=(3.5 * n_ckpts, 2.4 * n_batches),
        squeeze=False,
    )
    fig.suptitle(
        f"Run {run_key} ({run_id}) | noise_var={NOISE_VARIANCE} | {title_suffix}",
        fontsize=12,
    )

    for row_idx in range(n_batches):
        true_y = final_y[row_idx].item()
        dmmse_y = dmmse_preds[row_idx].item()
        ridge_y = ridge_preds[row_idx].item()
        for col_idx, ckpt_idx in enumerate(ckpt_indices):
            ax = axes[row_idx][col_idx]
            prob_dist = probs_per_ckpt[col_idx][row_idx].detach().cpu().numpy()
            ax.hist(
                indices,
                weights=prob_dist,
                bins=len(indices),
                density=True,
                color="white",
                edgecolor="black",
            )
            model_mean_y = model_means_per_ckpt[col_idx][row_idx]
            ax.axvline(x=true_y, color="#FF0000", linestyle="-", label="True y", linewidth=1)
            ax.axvline(x=dmmse_y, color="#9933FF", linestyle="--", label="DMMSE", linewidth=1.5) 
            ax.axvline(x=ridge_y, color="#00CCFF", linestyle=":", label="Ridge", linewidth=1.5)
            ax.axvline(x=model_mean_y, color="#33CC33", linestyle="-.", label="Model Mean", linewidth=1.5)

            if row_idx == 0:
                ax.set_title(f"ckpt {ckpt_idx}")
            if col_idx == 0:
                ax.set_ylabel(f"Batch {row_idx} (y={true_y:.2f})")
            ax.set_xlabel("y value")
            ax.grid(True)
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Iterate over all configured runs and evaluate across checkpoints
for run_key, run_info in RUNS.items():
    run_id = run_info["run_id"]
    ckpt_indices = run_info["ckpts"]

    pretrain_path, general_path = _load_distribution_paths(run_id, run_info["task_size"])
    pretrain_task_distribution = load_task_distribution_from_pt(pretrain_path, device=device)
    general_task_distribution = load_task_distribution_from_pt(general_path, device=device)

    # Memorisation-style sampling: sample from pretrain distribution
    regression_sequence_distribution = RegressionSequenceDistribution(
        pretrain_task_distribution, noise_variance=NOISE_VARIANCE
    ).to(device)
    xs, ys = regression_sequence_distribution.get_batch(num_examples=64, batch_size=batch_size)
    xs = xs.to(device)
    ys = ys.to(device)

    dmmse_preds = dmmse_predictor(xs, ys, pretrain_task_distribution, 0.25)[:, -1, :].cpu() #hardcoding for now
    ridge_preds = ridge_predictor(xs, ys, 0.25)[:, -1, :].cpu()

    probs_per_ckpt = []
    model_means_per_ckpt = []
    for ckpt_idx in ckpt_indices:
        probs, model_mean = _compute_model_outputs_for_ckpt(run_id, ckpt_idx, xs, ys)
        probs_per_ckpt.append(probs)
        model_means_per_ckpt.append(model_mean)

    _plot_grid_across_checkpoints(
        run_key,
        run_id,
        ckpt_indices,
        xs,
        ys,
        dmmse_preds,
        ridge_preds,
        probs_per_ckpt,
        model_means_per_ckpt,
        title_suffix=f"task_size={run_info['task_size']} | pretrain-sampled",
    )

    # Generalisation-style sampling: sample from true/general distribution
    regression_sequence_distribution = RegressionSequenceDistribution(
        general_task_distribution, noise_variance=NOISE_VARIANCE
    ).to(device)
    xs, ys = regression_sequence_distribution.get_batch(num_examples=64, batch_size=batch_size)
    xs = xs.to(device)
    ys = ys.to(device)

    # DMMSE and Ridge remain based on pretrain task distribution as before
    dmmse_preds = dmmse_predictor(xs, ys, pretrain_task_distribution, 0.25)[:, -1, :].cpu()
    ridge_preds = ridge_predictor(xs, ys, 0.25)[:, -1, :].cpu()

    probs_per_ckpt = []
    model_means_per_ckpt = []
    for ckpt_idx in ckpt_indices:
        probs, model_mean = _compute_model_outputs_for_ckpt(run_id, ckpt_idx, xs, ys)
        probs_per_ckpt.append(probs)
        model_means_per_ckpt.append(model_mean)

    _plot_grid_across_checkpoints(
        run_key,
        run_id,
        ckpt_indices,
        xs,
        ys,
        dmmse_preds,
        ridge_preds,
        probs_per_ckpt,
        model_means_per_ckpt,
        title_suffix=f"task_size={run_info['task_size']} | generalisation-sampled",
    )

# %%
