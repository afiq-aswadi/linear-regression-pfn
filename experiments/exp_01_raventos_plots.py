"""
Given checkpoints from raventos_train.py, we replicate the plots from the raventos paper.
"""


#%% Imports & configuration
import os
import re
from typing import Dict, List, Tuple

import torch

from models.model import AutoregressivePFN
from models.model_config import ModelConfig
from samplers.tasks import RegressionSequenceDistribution
from evals import ICLEvaluator
from raventos_train import load_task_distribution_from_pt, plot_results


#%% Paths and constants
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", os.path.join(PROJECT_ROOT, "checkpoints"))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Only consider runs at or after this run id (inclusive)
MIN_RUN_ID = os.environ.get("MIN_RUN_ID", "")

# Evaluation / model hyperparameters (matching raventos_train.py)
EVAL_NUM_EXAMPLES = 64
EVAL_BATCH_SIZE = 1024
NOISE_VAR = 0.25

# Model config used during those runs (see raventos_train.py main())
MODEL_CFG = ModelConfig(
    d_model=64,
    d_x=2,
    d_y=1,
    n_layers=2,
    n_heads=2,
    d_mlp=4 * 64,
    d_vocab=64,
    n_ctx=2 * EVAL_NUM_EXAMPLES,
)


#%% Discover available runs and final checkpoints
def discover_runs(checkpoints_dir: str, min_run_id: str) -> List[Dict]:
    model_ckpt_re = re.compile(r"^(?P<run>\d{8}_\d{6})_model_checkpoint_(?P<i>\d+)\\.pt$")
    pretrain_re = re.compile(r"^(?P<run>\d{8}_\d{6})_pretrain_discrete_(?P<num>\d+)tasks_(?P<d>\d+)d\\.pt$")
    true_re = re.compile(r"^(?P<run>\d{8}_\d{6})_true_gaussian_(?P<d>\d+)d\\.pt$")

    runs: Dict[str, Dict] = {}
    # Debug: show directory contents to help diagnose path issues
    try:
        print(f"Scanning checkpoints in: {checkpoints_dir}")
        files = os.listdir(checkpoints_dir)
        print(f"Found {len(files)} files")
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    matched_ckpt = matched_pretrain = matched_true = 0
    for fname in os.listdir(checkpoints_dir):
        m = model_ckpt_re.match(fname)
        if m:
            run_id = m.group("run")
            if run_id < min_run_id:
                continue
            runs.setdefault(run_id, {"model_ckpts": [], "pretrain": None, "true": None})
            runs[run_id]["model_ckpts"].append(os.path.join(checkpoints_dir, fname))
            matched_ckpt += 1
            continue

        m = pretrain_re.match(fname)
        if m:
            run_id = m.group("run")
            if run_id < min_run_id:
                continue
            runs.setdefault(run_id, {"model_ckpts": [], "pretrain": None, "true": None})
            runs[run_id]["pretrain"] = os.path.join(checkpoints_dir, fname)
            runs[run_id]["num_tasks"] = int(m.group("num"))
            runs[run_id]["task_size"] = int(m.group("d"))
            matched_pretrain += 1
            continue

        m = true_re.match(fname)
        if m:
            run_id = m.group("run")
            if run_id < min_run_id:
                continue
            runs.setdefault(run_id, {"model_ckpts": [], "pretrain": None, "true": None})
            runs[run_id]["true"] = os.path.join(checkpoints_dir, fname)
            runs[run_id]["task_size_true"] = int(m.group("d"))
            matched_true += 1
            continue

    # Reduce to runs that have at least one model checkpoint and both distributions
    run_infos: List[Dict] = []
    for run_id, info in runs.items():
        if not info["model_ckpts"] or info["pretrain"] is None or info["true"] is None:
            continue
        # pick highest index checkpoint
        def ckpt_index(path: str) -> int:
            return int(os.path.splitext(path)[0].split("_")[-1])

        final_ckpt = max(info["model_ckpts"], key=ckpt_index)
        run_infos.append({
            "run_id": run_id,
            "final_ckpt": final_ckpt,
            "pretrain": info["pretrain"],
            "true": info["true"],
            "num_tasks": info.get("num_tasks"),
            "task_size": info.get("task_size"),
            "task_size_true": info.get("task_size_true"),
        })

    run_infos.sort(key=lambda r: r["run_id"])  # chronological
    print(f"Matched: ckpts={matched_ckpt}, pretrain={matched_pretrain}, true={matched_true}")
    return run_infos


#%% Fallback: recursively search anywhere under project root
def discover_runs_recursive(project_root: str, min_run_id: str) -> List[Dict]:
    model_ckpt_re = re.compile(r"^(?P<run>\d{8}_\d{6})_model_checkpoint_(?P<i>\d+)\.pt$")
    pretrain_re = re.compile(r"^(?P<run>\d{8}_\d{6})_pretrain_discrete_(?P<num>\d+)tasks_(?P<d>\d+)d\.pt$")
    true_re = re.compile(r"^(?P<run>\d{8}_\d{6})_true_gaussian_(?P<d>\d+)d\.pt$")

    runs: Dict[str, Dict] = {}
    scanned_files = 0
    for root, _dirs, files in os.walk(project_root):
        for fname in files:
            scanned_files += 1
            m = model_ckpt_re.match(fname)
            if m:
                run_id = m.group("run")
                if min_run_id and run_id < min_run_id:
                    continue
                runs.setdefault(run_id, {"model_ckpts": [], "pretrain": None, "true": None})
                runs[run_id]["model_ckpts"].append(os.path.join(root, fname))
                continue

            m = pretrain_re.match(fname)
            if m:
                run_id = m.group("run")
                if min_run_id and run_id < min_run_id:
                    continue
                runs.setdefault(run_id, {"model_ckpts": [], "pretrain": None, "true": None})
                runs[run_id]["pretrain"] = os.path.join(root, fname)
                runs[run_id]["num_tasks"] = int(m.group("num"))
                runs[run_id]["task_size"] = int(m.group("d"))
                continue

            m = true_re.match(fname)
            if m:
                run_id = m.group("run")
                if min_run_id and run_id < min_run_id:
                    continue
                runs.setdefault(run_id, {"model_ckpts": [], "pretrain": None, "true": None})
                runs[run_id]["true"] = os.path.join(root, fname)
                runs[run_id]["task_size_true"] = int(m.group("d"))
                continue

    run_infos: List[Dict] = []
    for run_id, info in runs.items():
        if not info["model_ckpts"] or info["pretrain"] is None or info["true"] is None:
            continue
        def ckpt_index(path: str) -> int:
            return int(os.path.splitext(path)[0].split("_")[-1])
        final_ckpt = max(info["model_ckpts"], key=ckpt_index)
        run_infos.append({
            "run_id": run_id,
            "final_ckpt": final_ckpt,
            "pretrain": info["pretrain"],
            "true": info["true"],
            "num_tasks": info.get("num_tasks"),
            "task_size": info.get("task_size"),
            "task_size_true": info.get("task_size_true"),
        })

    run_infos.sort(key=lambda r: r["run_id"])  # chronological
    print(f"Recursive scan under {project_root} examined ~{scanned_files} files; found {len(run_infos)} runs")
    return run_infos


runs = discover_runs(CHECKPOINTS_DIR, MIN_RUN_ID)
print(f"Discovered {len(runs)} runs at/after {MIN_RUN_ID}")
for r in runs:
    print(f"- {r['run_id']}: ckpt={os.path.basename(r['final_ckpt'])}, pretrain={os.path.basename(r['pretrain'])}, true={os.path.basename(r['true'])}")

if len(runs) == 0 and MIN_RUN_ID:
    # Fallback: try without filtering
    print("No runs found with MIN_RUN_ID filter; retrying without filter...")
    runs = discover_runs(CHECKPOINTS_DIR, "")
    print(f"Discovered {len(runs)} runs with no filter")
    for r in runs:
        print(f"- {r['run_id']}: ckpt={os.path.basename(r['final_ckpt'])}, pretrain={os.path.basename(r['pretrain'])}, true={os.path.basename(r['true'])}")

if len(runs) == 0:
    # Final fallback: recursive search anywhere under project root
    runs = discover_runs_recursive(PROJECT_ROOT, MIN_RUN_ID)
    for r in runs:
        print(f"- {r['run_id']}: ckpt={r['final_ckpt']}, pretrain={r['pretrain']}, true={r['true']}")


#%% Load distributions and assert consistency
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

assert len(runs) > 0, "No eligible runs found. Ensure checkpoints exist and MIN_RUN_ID is correct."

# Load and validate true distributions are consistent
true_tds = [load_task_distribution_from_pt(r["true"], device=device) for r in runs]
first_true_td = true_tds[0]
for i, td in enumerate(true_tds[1:], start=1):
    assert getattr(td, "task_size", None) == getattr(first_true_td, "task_size", None), \
        f"True distribution task_size mismatch between runs: {runs[0]['run_id']} and {runs[i]['run_id']}"

# Validate pretrain distributions share the same task dimensionality across runs
pretrain_tds = [load_task_distribution_from_pt(r["pretrain"], device=device) for r in runs]
pretrain_task_sizes = [getattr(td, "task_size", None) for td in pretrain_tds]
assert len(set(pretrain_task_sizes)) == 1, "Pretrain task_size differs across runs."


#%% Evaluate each run with fixed evaluator
def evaluate_single_run(run: Dict, model_cfg: ModelConfig, device: str) -> Dict[str, float]:
    # Load distributions for this run
    pretrain_td = load_task_distribution_from_pt(run["pretrain"], device=device)
    true_td = load_task_distribution_from_pt(run["true"], device=device)

    pretrain_dist = RegressionSequenceDistribution(
        task_distribution=pretrain_td,
        noise_variance=NOISE_VAR,
    ).to(device)
    true_dist = RegressionSequenceDistribution(
        task_distribution=true_td,
        noise_variance=NOISE_VAR,
    ).to(device)

    # Build model and load final checkpoint
    model = AutoregressivePFN(model_cfg).to(device)
    state = torch.load(run["final_ckpt"], map_location=device)
    model.load_state_dict(state)
    model.eval()

    evaluator = ICLEvaluator(
        pretrain_dist=pretrain_dist,
        true_dist=true_dist,
        max_examples=EVAL_NUM_EXAMPLES,
        eval_batch_size=EVAL_BATCH_SIZE,
    )

    with torch.no_grad():
        metrics = evaluator(model)

    # Convert to plain floats and add extras
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

    clean_metrics["num_tasks"] = int(run.get("num_tasks") or 0)
    clean_metrics["task_size"] = int(run.get("task_size") or run.get("task_size_true") or 0)

    return clean_metrics


results: List[Dict[str, float]] = []
for run in runs:
    print(f"Evaluating run {run['run_id']} with final checkpoint {os.path.basename(run['final_ckpt'])}")
    res = evaluate_single_run(run, MODEL_CFG, device)
    results.append(res)


#%% Plot results
out_path = os.path.join(PLOTS_DIR, "raventos_experiment_fixed.png")
num_tasks_list = [r["num_tasks"] for r in runs]
plot_results(num_tasks_list, results, out_path)
print(f"Saved plot to: {out_path}")

# %%
