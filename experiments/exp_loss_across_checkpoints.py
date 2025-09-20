"""Plot mean-squared-error across checkpoints for each model run."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

if __package__ is None or __package__ == "":
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import torch

from samplers.tasks import RegressionSequenceDistribution
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


# Evaluation hyperparameters
NOISE_VARIANCE = 0.25
BATCH_SIZE = 64
PROMPT_LEN = 64
EVAL_SEED_BASE = 42

device = get_device()
model_config = RAVENTOS_SWEEP_MODEL_CONFIG
BASE_PLOT_DIR = ensure_experiment_dir(PLOTS_DIR, __file__)

def _set_seed(seed: int) -> None:
    """Seed torch (and CUDA if available) for deterministic sampling."""
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_dataset(
    sequence_dist: RegressionSequenceDistribution,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a single dataset using a fixed seed so checkpoints share examples."""
    _set_seed(seed)
    xs, ys = sequence_dist.get_batch(num_examples=PROMPT_LEN, batch_size=BATCH_SIZE)
    return xs.to(device), ys.to(device)


def _compute_mse(model, xs: torch.Tensor, ys: torch.Tensor) -> float:
    """Return mean-squared-error between model means and true targets."""
    with torch.inference_mode():
        _, model_means = model.get_model_mean_prediction(xs, ys)
        mse = torch.mean((model_means - ys.squeeze(-1)) ** 2)
    return float(mse.item())


def _plot_losses(
    ckpt_steps: Iterable[int],
    series: Iterable[Tuple[str, Iterable[float]]],
    title: str,
    filename: str,
    output_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    steps_list = list(ckpt_steps)

    for label, losses in series:
        ax.plot(steps_list, list(losses), marker="o", label=label)

    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("mean squared error")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if steps_list:
        if len(steps_list) == 1:
            tick_positions = steps_list
        else:
            import numpy as np

            idx_ticks = np.linspace(
                0,
                len(steps_list) - 1,
                num=min(len(steps_list), 8),
            ).round().astype(int)
            tick_positions = [steps_list[i] for i in idx_ticks]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(int(tick)) for tick in tick_positions], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not RUNS:
        raise ValueError("No runs configured in experiments.experiment_configs.RUNS")

    for idx, (run_key, run_info) in enumerate(RUNS.items()):
        run_id = run_info["run_id"]
        num_tasks = run_info["task_size"]
        ckpt_indices: List[int] = run_info["ckpts"]

        run_output_dir = BASE_PLOT_DIR  # Save all plots for this experiment in a single directory

        pretrain_path = get_pretrain_distribution_path(
            CHECKPOINTS_DIR,
            run_id,
            num_tasks=num_tasks,
            task_size=model_config.d_x,
        )
        general_path = get_true_distribution_path(
            CHECKPOINTS_DIR,
            run_id,
            task_size=model_config.d_x,
        )

        pretrain_dist = load_task_distribution(pretrain_path, device=device)
        general_dist = load_task_distribution(general_path, device=device)

        pretrain_seq = RegressionSequenceDistribution(
            pretrain_dist, noise_variance=NOISE_VARIANCE
        ).to(device)
        general_seq = RegressionSequenceDistribution(
            general_dist, noise_variance=NOISE_VARIANCE
        ).to(device)

        seed_offset = EVAL_SEED_BASE + idx * 10
        pretrain_xs, pretrain_ys = _sample_dataset(pretrain_seq, seed_offset)
        general_xs, general_ys = _sample_dataset(general_seq, seed_offset + 1)

        available_steps: List[int] = []
        pretrain_losses: List[float] = []
        general_losses: List[float] = []

        for ckpt_idx in ckpt_indices:
            ckpt_path = build_checkpoint_path(
                CHECKPOINTS_DIR,
                run_id,
                num_tasks=num_tasks,
                ckpt_idx=ckpt_idx,
            )
            if not os.path.exists(ckpt_path):
                print(f"[warn] Missing checkpoint skipped: {ckpt_path}")
                continue

            model = load_model_from_checkpoint(model_config, ckpt_path, device=device)

            pretrain_losses.append(_compute_mse(model, pretrain_xs, pretrain_ys))
            general_losses.append(_compute_mse(model, general_xs, general_ys))
            available_steps.append(ckpt_idx)

        if not available_steps:
            print(f"[warn] No checkpoints found for run {run_key} ({run_id}). Skipping plots.")
            continue

        plot_title = (
            f"Run {run_key} ({run_id}) | tasks={num_tasks} | pretrain vs general"
        )

        plot_filename = build_experiment_filename(
            "loss-curve",
            run=run_key,
            run_id=run_id,
            tasks=num_tasks,
            dataset="pretrain-generalisation",
            prompt_len=PROMPT_LEN,
            batch_size=BATCH_SIZE,
            noise=NOISE_VARIANCE,
            ckpt_min=min(available_steps),
            ckpt_max=max(available_steps),
        )

        _plot_losses(
            available_steps,
            (
                ("pretraining", pretrain_losses),
                ("generalisation", general_losses),
            ),
            plot_title,
            plot_filename,
            run_output_dir,
        )

        print(
            f"Saved loss curves for run {run_key} ({run_id}) to {run_output_dir}: "
            f"combined -> {plot_filename}"
        )


if __name__ == "__main__":
    main()
