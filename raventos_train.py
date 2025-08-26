"""
Raventos et al. style experiment:

- Sweep the number of pretraining tasks (task diversity)
- Train a model for each setting using the existing training loop
- Reload the saved task distributions from checkpoints
- Evaluate deltas and MSEs using the evaluator
- Plot metrics vs number of pretraining tasks

Usage (defaults chosen to run quickly on CPU):
    python raventos_train.py

You can override defaults with flags, e.g.:
    python raventos_train.py --steps 10000 --batch 512 --tasks 4 16 64 256 1024 4096 65536

TODO: parallelise training.
"""

import argparse
import os
from typing import Dict, List, Tuple
import torch.multiprocessing as mp

import torch
import matplotlib.pyplot as plt

from train import train as train_once
from models.model_config import ModelConfig
from evals import ICLEvaluator
from samplers.tasks import (
    DiscreteTaskDistribution,
    GaussianTaskDistribution,
    RegressionSequenceDistribution,
)


def train_single_config(args_tuple):
    """Train a single model configuration on a specific GPU"""
    (num_tasks, task_size, model_cfg, base_training_cfg, device_id) = args_tuple
    
    # Set specific GPU
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device_id)
    
    training_cfg = dict(base_training_cfg)
    training_cfg["task_size"] = task_size
    training_cfg["num_tasks"] = int(num_tasks)
    training_cfg["device"] = device
    
    print(f"Training on GPU {device_id}: num_tasks={num_tasks}, task_size={task_size}")
    
    run_id, model, checkpoints_dir = train_once(
        config=model_cfg,
        training_config=training_cfg,
        print_model_dimensionality=False,
    )
    
    # Move back to CPU for evaluation to save GPU memory
    model = model.cpu()
    
    metrics = evaluate_run(
        run_id=run_id,
        checkpoints_dir=checkpoints_dir,
        model=model,
        task_size=task_size,
        noise_var=training_cfg["noise_var"],
        num_examples=training_cfg["num_examples"],
        eval_batch_size=training_cfg["eval_batch_size"],
        device="cpu",  # Eval on CPU
    )
    
    metrics["num_tasks"] = int(num_tasks)
    metrics["task_size"] = int(task_size)
    return metrics


def load_task_distribution_from_pt(file_path: str, device: str = "cpu"):
    """
    Load a task distribution saved by `samplers.tasks.*.save`.

    Returns a concrete TaskDistribution instance on the requested device.
    """
    state = torch.load(file_path, map_location=device)
    dist_type = state.get("type")
    if dist_type == "discrete":
        # Reconstruct and inject saved tasks
        td = DiscreteTaskDistribution(
            task_size=state["task_size"],
            num_tasks=state["num_tasks"],
            device=device,
        )
        td.tasks = state["tasks"].to(device)
        return td
    if dist_type == "gaussian":
        return GaussianTaskDistribution(task_size=state["task_size"], device=device)
    raise ValueError(f"Unrecognized task distribution type in {file_path}: {dist_type}")


def evaluate_run(
    run_id: str,
    checkpoints_dir: str,
    model,
    task_size: int,
    noise_var: float,
    num_examples: int,
    eval_batch_size: int,
    device: str,
) -> Dict[str, float]:
    """
    Build the exact pretrain/true distributions used during training from the
    saved files, wrap them as `RegressionSequenceDistribution`, and evaluate.
    """
    pretrain_path = os.path.join(
        checkpoints_dir, f"{run_id}_pretrain_discrete_*tasks_{task_size}d.pt"
    )
    true_path = os.path.join(
        checkpoints_dir, f"{run_id}_true_gaussian_{task_size}d.pt"
    )

    # Find concrete filenames (wildcard only for robustness w.r.t different num_tasks)
    # We intentionally avoid globbing external modules; do a tiny manual search.
    def resolve_single(pattern: str) -> str:
        directory = os.path.dirname(pattern)
        prefix = os.path.basename(pattern).split("*")[0]
        suffix = os.path.basename(pattern).split("*")[-1]
        candidates = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(suffix)]
        if not candidates:
            raise FileNotFoundError(f"No file matching {pattern}")
        if len(candidates) > 1:
            # Pick the latest modified file
            candidates.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))
        return os.path.join(directory, candidates[-1])

    pretrain_td_file = resolve_single(pretrain_path)
    true_td_file = resolve_single(true_path)

    pretrain_td = load_task_distribution_from_pt(pretrain_td_file, device=device)
    true_td = load_task_distribution_from_pt(true_td_file, device=device)

    pretrain_dist = RegressionSequenceDistribution(
        task_distribution=pretrain_td,
        noise_variance=noise_var,
    ).to(device)
    true_dist = RegressionSequenceDistribution(
        task_distribution=true_td,
        noise_variance=noise_var,
    ).to(device)

    evaluator = ICLEvaluator(
        pretrain_dist=pretrain_dist,
        true_dist=true_dist,
        max_examples=num_examples,
        eval_batch_size=eval_batch_size,
    )
    model.eval()
    with torch.no_grad():
        metrics = evaluator(model)
    model.train()

    # Ensure plain floats for downstream plotting/aggregation
    clean_metrics: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            try:
                clean_metrics[k] = float(v.item())
            except Exception:
                clean_metrics[k] = float(v.mean().item())
        else:
            clean_metrics[k] = float(v)

    # Add baseline MSEs (theoretical optima on each distribution)
    def mse_tensor(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        return float((y_true - y_pred).square().mean().item())

    clean_metrics["baseline/pretrain/mse_dmmse"] = mse_tensor(
        evaluator.pretrain_ys, evaluator.pretrain_dmmse_preds
    )
    clean_metrics["baseline/pretrain/mse_ridge"] = mse_tensor(
        evaluator.pretrain_ys, evaluator.pretrain_ridge_preds
    )
    clean_metrics["baseline/true/mse_dmmse"] = mse_tensor(
        evaluator.true_ys, evaluator.true_dmmse_preds
    )
    clean_metrics["baseline/true/mse_ridge"] = mse_tensor(
        evaluator.true_ys, evaluator.true_ridge_preds
    )
    return clean_metrics


def run_sweep(
    task_sizes: List[int],
    num_tasks_list: List[int],
    model_cfg: ModelConfig,
    base_training_cfg: Dict,
    device: str,
) -> Tuple[List[int], List[Dict[str, float]]]:
    """
    For each value of `num_tasks`, train a new model and evaluate metrics.
    Returns the list of num_tasks and corresponding metrics dicts.
    """
    results: List[Dict[str, float]] = []

    # We support sweeping over different task dimensionalities, though Raventos
    # commonly fixes task_size and varies the number of tasks. We'll run over
    # each task_size sequentially and aggregate metrics per num_tasks.
    for task_size in task_sizes:
        for num_tasks in num_tasks_list:
            training_cfg = dict(base_training_cfg)
            training_cfg["task_size"] = task_size
            training_cfg["num_tasks"] = int(num_tasks)

            print(f"\n=== Training with num_tasks={num_tasks}, task_size={task_size} ===")
            run_id, model, checkpoints_dir = train_once(
                config=model_cfg,
                training_config=training_cfg,
                print_model_dimensionality=False,
                plot_checkpoints=False,
            )

            print("Evaluating deltas and MSEs...")
            metrics = evaluate_run(
                run_id=run_id,
                checkpoints_dir=checkpoints_dir,
                model=model,
                task_size=task_size,
                noise_var=training_cfg["noise_var"],
                num_examples=training_cfg["num_examples"],
                eval_batch_size=training_cfg["eval_batch_size"],
                device=device,
            )
            metrics["num_tasks"] = int(num_tasks)
            metrics["task_size"] = int(task_size)
            results.append(metrics)

    return num_tasks_list, results


def run_sweep_parallel(
    task_sizes: List[int],
    num_tasks_list: List[int],
    model_cfg: ModelConfig,
    base_training_cfg: Dict,
    num_gpus: int = None,
) -> Tuple[List[int], List[Dict[str, float]]]:
    """
    Run the sweep in parallel across multiple GPUs.
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available for parallel training")
    
    print(f"Using {num_gpus} GPUs for parallel training")
    
    # Create all training configs
    configs = []
    for task_size in task_sizes:
        for i, num_tasks in enumerate(num_tasks_list):
            device_id = i % num_gpus  # Round-robin GPU assignment
            configs.append((num_tasks, task_size, model_cfg, base_training_cfg, device_id))
    
    # Run in parallel using spawn start method (CUDA-safe)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_gpus) as pool:
        results = pool.map(train_single_config, configs)
    
    return num_tasks_list, results


def plot_results(num_tasks_list: List[int], results: List[Dict[str, float]], out_path: str):
    """
    Produce a compact figure showing memorisation -> generalisation trends.
    """
    # Sort results by num_tasks
    results = sorted(results, key=lambda r: r["num_tasks"])

    xs = [r["num_tasks"] for r in results]
    pretrain_mse = [r["mse/pretrain"] for r in results]
    true_mse = [r["mse/true"] for r in results]
    delta_pretrain_dmmse = [r["deltas/pretrain/delta_dmmse"] for r in results]
    delta_true_ridge = [r["deltas/true/delta_ridge"] for r in results]

    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

    # Top row: Prompt from pretraining distribution
    # (Left) MSE/D with baselines
    axs[0, 0].plot(xs, pretrain_mse, marker="o", label="Model")
    axs[0, 0].plot(xs, [r["baseline/pretrain/mse_dmmse"] for r in results], linestyle="--", label="dMMSE (theory)")
    axs[0, 0].plot(xs, [r["baseline/pretrain/mse_ridge"] for r in results], linestyle=":", label="Ridge (theory)")
    axs[0, 0].set_xscale("log", base=2)
    axs[0, 0].set_ylabel("MSE/D")
    axs[0, 0].set_title("Pretrain prompts: MSE/D")
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls=":", alpha=0.5)

    # (Middle) Δ_PT,dMMSE
    axs[0, 1].plot(xs, delta_pretrain_dmmse, marker="o")
    axs[0, 1].set_xscale("log", base=2)
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_title("Pretrain prompts: Δ_PT,dMMSE")
    axs[0, 1].grid(True, which="both", ls=":", alpha=0.5)

    # (Right) Δ_PT,Ridge on pretrain prompts for symmetry/inspection
    axs[0, 2].plot(xs, [r["deltas/pretrain/delta_ridge"] for r in results], marker="o")
    axs[0, 2].set_xscale("log", base=2)
    axs[0, 2].set_yscale("log")
    axs[0, 2].set_title("Pretrain prompts: Δ_PT,Ridge")
    axs[0, 2].grid(True, which="both", ls=":", alpha=0.5)

    # Bottom row: Prompt from true Gaussian distribution
    # (Left) MSE/D with baselines
    axs[1, 0].plot(xs, true_mse, marker="o", label="Model")
    axs[1, 0].plot(xs, [r["baseline/true/mse_dmmse"] for r in results], linestyle="--", label="dMMSE (theory)")
    axs[1, 0].plot(xs, [r["baseline/true/mse_ridge"] for r in results], linestyle=":", label="Ridge (theory)")
    axs[1, 0].set_xscale("log", base=2)
    axs[1, 0].set_xlabel("# Pretraining Tasks")
    axs[1, 0].set_ylabel("MSE/D")
    axs[1, 0].set_title("True prompts: MSE/D")
    axs[1, 0].legend()
    axs[1, 0].grid(True, which="both", ls=":", alpha=0.5)

    # (Middle) Δ_PT,dMMSE for true prompts
    axs[1, 1].plot(xs, [r["deltas/true/delta_dmmse"] for r in results], marker="o")
    axs[1, 1].set_xscale("log", base=2)
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlabel("# Pretraining Tasks")
    axs[1, 1].set_title("True prompts: Δ_PT,dMMSE")
    axs[1, 1].grid(True, which="both", ls=":", alpha=0.5)

    # (Right) Δ_PT,Ridge for true prompts
    axs[1, 2].plot(xs, delta_true_ridge, marker="o")
    axs[1, 2].set_xscale("log", base=2)
    axs[1, 2].set_yscale("log")
    axs[1, 2].set_xlabel("# Pretraining Tasks")
    axs[1, 2].set_title("True prompts: Δ_PT,Ridge")
    axs[1, 2].grid(True, which="both", ls=":", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raventos-style diversity sweep experiment")
    parser.add_argument(
        "--tasks",
        nargs="*",
        type=int,
        default=[2**i for i in range(16)],
        help="List of numbers of pretraining tasks to sweep",
    )
    parser.add_argument(
        "--task_size",
        nargs="*",
        type=int,
        default=[16], #todo: fix
        help="List of task dimensionalities to sweep (usually fixed)",
    )
    parser.add_argument("--steps", type=int, default=500000, help="Training steps per run")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size")
    parser.add_argument("--eval_batch", type=int, default=256, help="Eval batch size")
    parser.add_argument("--examples", type=int, default=64, help="# in-context examples per sequence")
    parser.add_argument("--noise", type=float, default=0.25, help="Output noise variance")
    parser.add_argument("--lr", type=float, default=1e-3, help="Max learning rate")
    parser.add_argument("--checkpoints", type=int, default=10, help="# of checkpoints to save per run (None to disable)")
    parser.add_argument("--parallel", action="store_true", help="Enable multi-GPU parallel training")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use (default: all available)")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "plots", "raventos_experiment.png"),
        help="Path to save output figure",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Model architecture (keep modest for the sweep)
    model_config = ModelConfig(
        d_model=128,
        d_x=16,
        d_y=1,
        n_layers=2,
        n_heads=2,
        d_mlp=256,
        d_vocab=128,
        n_ctx=2 * args.examples,
        d_head = 64,
        y_min=-7,
        y_max=7,
    )

    base_training_config = {
        "device": device,
        "task_size": args.task_size[0],  # placeholder; overwritten in sweep
        "num_tasks": args.tasks[0],      # placeholder; overwritten in sweep
        "noise_var": args.noise,
        "num_examples": args.examples,
        "learning_rate": args.lr,
        "training_steps": args.steps,
        "batch_size": args.batch,
        "eval_batch_size": args.eval_batch,
        "print_loss_interval": max(50, args.steps // 50),
        "print_metrics_interval": max(200, args.steps // 10),
        "n_checkpoints": args.checkpoints,
        "logarithmic_checkpoints": True,
    }

    # Run sweep in parallel or serial mode
    if args.parallel and torch.cuda.device_count() > 1:
        print("Running in parallel mode")
        num_tasks_list, results = run_sweep_parallel(
            task_sizes=args.task_size,
            num_tasks_list=args.tasks,
            model_cfg=model_config,
            base_training_cfg=base_training_config,
            num_gpus=args.gpus
        )
    else:
        if args.parallel:
            print("Warning: Parallel mode requested but only 1 or 0 GPUs available. Running in serial mode.")
        print("Running in serial mode")
        num_tasks_list, results = run_sweep(
            task_sizes=args.task_size,
            num_tasks_list=args.tasks,
            model_cfg=model_config,
            base_training_cfg=base_training_config,
            device=device,
        )

    plot_results(num_tasks_list, results, args.out)


if __name__ == "__main__":
    main()


