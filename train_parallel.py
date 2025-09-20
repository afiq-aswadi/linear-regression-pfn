"""
Raventos et al. style experiment:

- Sweep the number of pretraining tasks (task diversity)
- Train a model for each setting using the existing training loop
- Reload the saved task distributions from checkpoints
- Evaluate deltas and MSEs using the evaluator
- Log metrics for both pretraining and true task distributions

Usage (defaults chosen to run quickly on CPU):
    uv run raventos_train.py

You can override defaults with flags, e.g.:
    uv run raventos_train.py --steps 10000 --batch 512 --tasks 4 16 64 256 1024 4096 65536

"""

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional
import torch.multiprocessing as mp
import logging

import torch

from train import train as train_once
from models.model_config import ModelConfig
from evals import ICLEvaluator
from samplers.tasks import (
    DiscreteTaskDistribution,
    GaussianTaskDistribution,
    RegressionSequenceDistribution,
    load_task_distribution_from_pt,
)


logger = logging.getLogger(__name__)


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
    
    logger.info(
        "Starting training on device %s | num_tasks=%s task_size=%s",
        device,
        num_tasks,
        task_size,
    )
    
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
    metrics.setdefault("run_id", run_id)
    return metrics


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

    # Ensure plain floats for downstream logging/aggregation
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

            logger.info(
                "Running training sweep | num_tasks=%s task_size=%s",
                num_tasks,
                task_size,
            )
            run_id, model, checkpoints_dir = train_once(
                config=model_cfg,
                training_config=training_cfg,
                print_model_dimensionality=False,
                plot_checkpoints=False,
            )

            logger.info("Evaluating trained model | run_id=%s", run_id)
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
            metrics.setdefault("run_id", run_id)
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
    
    logger.info("Using %d GPUs for parallel training", num_gpus)
    
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


def _build_log_records(metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Split evaluator metrics into per-dataset dictionaries for logging."""
    return {
        "pretrain": {
            "mse": metrics.get("mse/pretrain"),
            "delta_dmmse": metrics.get("deltas/pretrain/delta_dmmse"),
            "delta_ridge": metrics.get("deltas/pretrain/delta_ridge"),
            "baseline_mse_dmmse": metrics.get("baseline/pretrain/mse_dmmse"),
            "baseline_mse_ridge": metrics.get("baseline/pretrain/mse_ridge"),
        },
        "true": {
            "mse": metrics.get("mse/true"),
            "delta_dmmse": metrics.get("deltas/true/delta_dmmse"),
            "delta_ridge": metrics.get("deltas/true/delta_ridge"),
            "baseline_mse_dmmse": metrics.get("baseline/true/mse_dmmse"),
            "baseline_mse_ridge": metrics.get("baseline/true/mse_ridge"),
        },
    }


def log_results(results: List[Dict[str, float]], log_path: Optional[str] = None) -> None:
    """Log evaluation metrics to stdout and optionally persist as JSON lines."""
    file_records = []
    for metrics in sorted(results, key=lambda r: (r["task_size"], r["num_tasks"])):
        dataset_summaries = _build_log_records(metrics)
        for dataset_name, dataset_values in dataset_summaries.items():
            safe_values = {
                key: (float(value) if value is not None else float("nan"))
                for key, value in dataset_values.items()
            }
            logger.info(
                "eval_result dataset=%s num_tasks=%s task_size=%s mse=%.6f delta_dmmse=%.6f delta_ridge=%.6f baseline_dmmse=%.6f baseline_ridge=%.6f",
                dataset_name,
                metrics["num_tasks"],
                metrics["task_size"],
                safe_values.get("mse"),
                safe_values.get("delta_dmmse"),
                safe_values.get("delta_ridge"),
                safe_values.get("baseline_mse_dmmse"),
                safe_values.get("baseline_mse_ridge"),
            )

            record = {
                "dataset": dataset_name,
                "num_tasks": metrics["num_tasks"],
                "task_size": metrics["task_size"],
                "metrics": dataset_values,
            }
            if "run_id" in metrics:
                record["run_id"] = metrics["run_id"]
            file_records.append(record)

    if log_path:
        directory = os.path.dirname(log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            for record in file_records:
                handle.write(json.dumps(record) + "\n")


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
        "--log_file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "logs", "raventos_eval.jsonl"),
        help="Optional JSONL file to append evaluation metrics to",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

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
        logger.info("Running in parallel mode")
        num_tasks_list, results = run_sweep_parallel(
            task_sizes=args.task_size,
            num_tasks_list=args.tasks,
            model_cfg=model_config,
            base_training_cfg=base_training_config,
            num_gpus=args.gpus
        )
    else:
        if args.parallel:
            logger.warning(
                "Parallel mode requested but only 1 or 0 GPUs available. Running in serial mode."
            )
        logger.info("Running in serial mode")
        num_tasks_list, results = run_sweep(
            task_sizes=args.task_size,
            num_tasks_list=args.tasks,
            model_cfg=model_config,
            base_training_cfg=base_training_config,
            device=device,
        )

    log_results(results, args.log_file)


if __name__ == "__main__":
    main()
