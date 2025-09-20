"""
Across checkpoints, plot model's predictive distributions from either true or general distributions.


"""
#%%

import os
import torch

from models.model import AutoregressivePFN
from samplers.tasks import load_task_distribution_from_pt, RegressionSequenceDistribution
from baselines import dmmse_predictor, ridge_predictor
from experiments.experiment_utils import (
    get_device,
    build_checkpoint_path,
    get_pretrain_distribution_path,
    get_true_distribution_path,
    load_model_from_checkpoint,
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
# m10_run = dict([("m12", RUNS["m12"])])
# m10_run["m12"]["ckpts"] = [m10_run["m12"]["ckpts"][-1]]


device = get_device()
model_config = RAVENTOS_SWEEP_MODEL_CONFIG
indices = torch.linspace(model_config.y_min, model_config.y_max, model_config.d_vocab)

#%%
batch_size = 5 #number of batches to plot
NOISE_VARIANCE = 0.25
prompt_len = 20

#%%
# Use a single noise variance for all runs; included in plot titles NOTE: These are sampling from true distribution.

import matplotlib.pyplot as plt
import torch.nn.functional as F


def _load_distribution_paths(run_id: str, num_tasks: int, task_size: int):
    pretrain_path = get_pretrain_distribution_path(CHECKPOINTS_DIR, run_id, num_tasks, task_size)
    general_path = get_true_distribution_path(CHECKPOINTS_DIR, run_id, task_size)
    return pretrain_path, general_path


def _compute_model_outputs_for_ckpt(num_tasks: int, run_id: str, ckpt_idx: int, xs: torch.Tensor, ys: torch.Tensor):
    model_path = build_checkpoint_path(CHECKPOINTS_DIR, run_id, num_tasks, ckpt_idx)
    model = load_model_from_checkpoint(model_config, model_path, device=device)

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
    output_dir: str,
    filename: str,
    save_plot: bool = True,
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
                density=False,
                color="white",
                edgecolor="black",
            )
            model_mean_y = model_means_per_ckpt[col_idx][row_idx]
            ax.axvline(x=true_y, color="#FF0000", linestyle="-", label="True y", linewidth=1)
            ax.axvline(x=dmmse_y, color="#9933FF", linestyle="--", label="DMMSE", linewidth=1.5) 
            ax.axvline(x=ridge_y, color="#00CCFF", linestyle=":", label="Ridge", linewidth=1.5)
            ax.axvline(x=model_mean_y, color="#33CC33", linestyle="-.", label="Model Mean", linewidth=1.5)

            if row_idx == 0:
                ax.set_title(f"step {ckpt_idx}")
            if col_idx == 0:
                ax.set_ylabel(f"Batch {row_idx} (y={true_y:.2f})")
            ax.set_xlabel("y value")
            ax.grid(True)
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_plot:
        plt.savefig(os.path.join(output_dir, filename))
    else:
        plt.show()


#%%
# Iterate over all configured runs and evaluate across checkpoints
for run_key, run_info in RUNS.items():
    run_id = run_info["run_id"]
    ckpt_indices = run_info["ckpts"]
    run_output_dir = ensure_experiment_dir(PLOTS_DIR, __file__, run_key)

    pretrain_path, general_path = _load_distribution_paths(run_id, num_tasks=run_info["task_size"], task_size=RAVENTOS_SWEEP_MODEL_CONFIG.d_x)
    pretrain_task_distribution = load_task_distribution_from_pt(pretrain_path, device=device)
    general_task_distribution = load_task_distribution_from_pt(general_path, device=device)

    # Memorisation-style sampling: sample from pretrain distribution
    regression_sequence_distribution = RegressionSequenceDistribution(
        pretrain_task_distribution, noise_variance=NOISE_VARIANCE
    ).to(device)
    xs, ys = regression_sequence_distribution.get_batch(num_examples=prompt_len, batch_size=batch_size)
    xs = xs.to(device)
    ys = ys.to(device)

    dmmse_preds = dmmse_predictor(xs, ys, pretrain_task_distribution, 0.25, bound_by_y_min_max=True, y_min=model_config.y_min, y_max=model_config.y_max)[:, -1, :].cpu() #hardcoding for now
    ridge_preds = ridge_predictor(xs, ys, 0.25, bound_by_y_min_max=True, y_min=model_config.y_min, y_max=model_config.y_max)[:, -1, :].cpu()

    probs_per_ckpt = []
    model_means_per_ckpt = []
    for ckpt_idx in ckpt_indices:
        probs, model_mean = _compute_model_outputs_for_ckpt(run_info["task_size"], run_id, ckpt_idx, xs, ys)
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
        title_suffix=f"num_tasks={run_info['task_size']} | prompt_len={prompt_len} | pretrain-sampled",
        output_dir=run_output_dir,
        filename=build_experiment_filename(
            "predictives-grid",
            run=run_key,
            run_id=run_id,
            tasks=run_info["task_size"],
            prompt_len=prompt_len,
            sample="pretrain",
            noise=NOISE_VARIANCE,
            ckpt_min=min(ckpt_indices),
            ckpt_max=max(ckpt_indices),
            batches=batch_size,
        ),
    )

    # Generalisation-style sampling: sample from true/general distribution
    regression_sequence_distribution = RegressionSequenceDistribution(
        general_task_distribution, noise_variance=NOISE_VARIANCE
    ).to(device)
    xs, ys = regression_sequence_distribution.get_batch(num_examples=prompt_len, batch_size=batch_size)
    xs = xs.to(device)
    ys = ys.to(device)

    # DMMSE and Ridge remain based on pretrain task distribution as before
    dmmse_preds = dmmse_predictor(xs, ys, pretrain_task_distribution, 0.25, bound_by_y_min_max=True, y_min=model_config.y_min, y_max=model_config.y_max)[:, -1, :].cpu()
    ridge_preds = ridge_predictor(xs, ys, 0.25, bound_by_y_min_max=True, y_min=model_config.y_min, y_max=model_config.y_max)[:, -1, :].cpu()

    probs_per_ckpt = []
    model_means_per_ckpt = []
    for ckpt_idx in ckpt_indices:
        probs, model_mean = _compute_model_outputs_for_ckpt(run_info["task_size"], run_id, ckpt_idx, xs, ys)
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
        title_suffix=f"num_tasks={run_info['task_size']} | prompt_len={prompt_len} | generalisation-sampled",
        output_dir=run_output_dir,
        filename=build_experiment_filename(
            "predictives-grid",
            run=run_key,
            run_id=run_id,
            tasks=run_info["task_size"],
            prompt_len=prompt_len,
            sample="generalisation",
            noise=NOISE_VARIANCE,
            ckpt_min=min(ckpt_indices),
            ckpt_max=max(ckpt_indices),
            batches=batch_size,
        ),
    )

# %%
