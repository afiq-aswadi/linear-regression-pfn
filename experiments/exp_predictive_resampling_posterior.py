"""
This script performs predictive resampling without conditioning on any input data across different models and checkpoints.
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
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import csv

import pandas as pd  

from models.model import AutoregressivePFN
from models.config import ModelConfig
from samplers.tasks import load_task_distribution_from_pt, RegressionSequenceDistribution
from predictive_resampling.predictive_resampling import predictive_resampling_beta_chunked, predictive_resampling_beta
from baselines import dmmse_w_hat, ridge_w_hat

from experiments.experiment_utils import (
    get_device,
    build_checkpoint_path,
    get_pretrain_distribution_path,
    get_true_distribution_path,
    load_model_from_checkpoint,
    get_model_codelength,
    get_ridge_codelength,
    load_task_distribution,
    extract_w_pool,
    build_experiment_filename,
    ensure_experiment_dir,
)
from experiments.experiment_configs import (
    RAVENTOS_SWEEP_MODEL_CONFIG,
    RUNS,
    CHECKPOINTS_DIR,
    PLOTS_DIR
)

def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the energy distance between two samples x and y.
    Inputs:
        x: shape (n, d)
        y: shape (m, d)
    Output:
        scalar energy distance
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n, m = len(x), len(y)

    d_xy = cdist(x, y, metric='euclidean')
    d_xx = cdist(x, x, metric='euclidean')
    d_yy = cdist(y, y, metric='euclidean')

    ed_squared = 2 * d_xy.mean() - d_xx.mean() - d_yy.mean()
    return np.sqrt(max(ed_squared, 0))  # numerical stability




#%%
device = get_device()
model_config = RAVENTOS_SWEEP_MODEL_CONFIG
BASE_PLOT_DIR = ensure_experiment_dir(PLOTS_DIR, __file__)
SUMMARY_PLOT_DIR = ensure_experiment_dir(PLOTS_DIR, __file__, "summary")
# Configuration
forward_recursion_steps = 64
forward_recursion_samples = 1000

#%%
TEST_SEED = 42
rng = np.random.default_rng(TEST_SEED)

N_test = 10
PROMPT_LEN = 8
D = 16
SIGMA2 = 0.25

X_test_all = rng.standard_normal(size=(N_test, PROMPT_LEN, D)).astype(np.float32)
w_true_all = rng.standard_normal(size=(N_test, D)).astype(np.float32)
noise = rng.normal(0, np.sqrt(SIGMA2), size=(N_test, PROMPT_LEN)).astype(np.float32)
y_test_all = np.einsum('tkd,td->tk', X_test_all, w_true_all) + noise

X_test_all = torch.from_numpy(X_test_all).to(device)
w_true_all = torch.from_numpy(w_true_all).to(device)
y_test_all = torch.from_numpy(y_test_all).unsqueeze(-1).to(device)


# Extract first two runs
# RUNS = dict(list(RUNS.items())[:2])




#%%
# Iterate through models and checkpoints
# Storage for Energy Distance metrics
results_full = {}
ed_records = []  # optional flat records for CSV/DataFrame

for run_key, run_info in RUNS.items():
    print(f"\nProcessing {run_key} (task_size={run_info['task_size']})...")
    # Get every 4th checkpoint
    selected_checkpoints = run_info['ckpts'][::4]
    print(f"Selected checkpoints: {selected_checkpoints}")

    # Load task distribution for this model
    try:
        pretrain_dist_path = get_pretrain_distribution_path(CHECKPOINTS_DIR, run_info["run_id"], run_info["task_size"], task_size = D)
        task_distribution = load_task_distribution(pretrain_dist_path, device=device)
        print(f"Loaded task distribution from {pretrain_dist_path}")
    except Exception as e:
        print(f"Warning: Could not load task distribution: {e}")
        task_distribution = None
 
    # Process each checkpoint
    for ckpt_idx, checkpoint_step in enumerate(selected_checkpoints):
        print(f"  Processing checkpoint {checkpoint_step}...")

    #Build checkpoint path and load model
        checkpoint_path = build_checkpoint_path(CHECKPOINTS_DIR, run_info["run_id"], run_info["task_size"], checkpoint_step)
        model = load_model_from_checkpoint(model_config, checkpoint_path, device=device)

        W_pool = task_distribution.tasks

        for test_idx in range(N_test):
            x_test = X_test_all[[test_idx]] # (K_TRAIN, D)
            y_test = y_test_all[[test_idx]] # (K_TRAIN, 1)
            w_true = w_true_all[[test_idx]] # (D,)

            beta_dmmse = dmmse_w_hat(x_test, y_test, task_distribution, noise_variance=SIGMA2).cpu().numpy()
            beta_ridge = ridge_w_hat(x_test, y_test, task_distribution, noise_variance=SIGMA2).cpu().numpy()

        # Perform predictive resampling
        beta_pt = predictive_resampling_beta_chunked(
            model, model_config, 
            forward_recursion_steps=forward_recursion_steps, 
            forward_recursion_samples=forward_recursion_samples,
            init_x = x_test,
            init_y = y_test
        )

         # --- Ridge posterior samples ---
        XtX = x_test.squeeze(0).T @ x_test.squeeze(0)
        A = XtX + SIGMA2 * torch.eye(D,device=device)
        Sigma_r = torch.linalg.inv(A)
        mu_r = Sigma_r @ x_test.squeeze(0).T @ y_test.squeeze(0)
        ridge_samples = np.random.multivariate_normal(mu_r[:,0].cpu().numpy(), Sigma_r.cpu().numpy(), size=forward_recursion_samples)

        # --- dMMSE posterior samples ---
        preds = x_test.squeeze(0) @ task_distribution.tasks.T  # shape (K, M)
        errs = (y_test.squeeze(0) - preds).T  # shape (M, K)
        log_w = -0.5 / SIGMA2 * (errs**2).sum(dim=1)
        weights = torch.softmax(log_w, dim=0)
        idxs = torch.multinomial(weights, forward_recursion_samples, replacement=True)
        dmmse_samples = W_pool[idxs].cpu().numpy()

        # --- Energy Distances ---
        ed_pt_dmmse = energy_distance(beta_pt, dmmse_samples)
        ed_pt_ridge = energy_distance(beta_pt, ridge_samples)
        ed_dmmse_ridge = energy_distance(dmmse_samples, ridge_samples)

        # print(f"  [M={run_info['task_size']} | step={checkpoint_step} | seq={test_idx}] "
        #       f"ED(PT,dMMSE)={ed_pt_dmmse:.4f} | ED(PT,ridge)={ed_pt_ridge:.4f} "
        #       f"| ED(dMMSE,ridge)={ed_dmmse_ridge:.4f} "
        #       f"| Closer to: {'dMMSE' if ed_pt_dmmse < ed_pt_ridge else 'ridge'}")

        # Aggregate metrics per run/step
        if run_key not in results_full:
            results_full[run_key] = {"M": run_info['task_size'], "by_step": {}}
        if checkpoint_step not in results_full[run_key]["by_step"]:
            results_full[run_key]["by_step"][checkpoint_step] = {"dmmse": [], "ridge": [], "baseline": []}
        results_full[run_key]["by_step"][checkpoint_step]["dmmse"].append(float(ed_pt_dmmse))
        results_full[run_key]["by_step"][checkpoint_step]["ridge"].append(float(ed_pt_ridge))
        results_full[run_key]["by_step"][checkpoint_step]["baseline"].append(float(ed_dmmse_ridge))

        # Flat record for optional DataFrame/CSV
        ed_records.append({
            "run_key": run_key,
            "M": run_info['task_size'],
            "step": int(checkpoint_step),
            "seq": int(test_idx),
            "ed_pt_dmmse": float(ed_pt_dmmse),
            "ed_pt_ridge": float(ed_pt_ridge),
            "ed_dmmse_ridge": float(ed_dmmse_ridge),
        })


 

print("\nPredictive resampling analysis complete!")

#%%

# Post-process: per-run plots and CSV export
if len(results_full) > 0:

    # Optional combined CSV across runs
    try:
        combined_csv_filename = build_experiment_filename(
            "energy-distance",
            extension="csv",
            recursion_steps=forward_recursion_steps,
            samples=forward_recursion_samples,
            n_test=N_test,
            prompt_len=PROMPT_LEN,
        )
        combined_csv_path = os.path.join(SUMMARY_PLOT_DIR, combined_csv_filename)
        with open(combined_csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_key", "M", "step", "seq",
                    "ed_pt_dmmse", "ed_pt_ridge", "ed_dmmse_ridge",
                ],
            )
            writer.writeheader()
            for row in ed_records:
                writer.writerow(row)
        print(f"Saved combined ED CSV: {combined_csv_path}")
    except Exception as e:
        print(f"Warning: failed to write combined ED CSV: {e}")

    # Per-run aggregation and plotting
    for run_key, run_data in results_full.items():
        M_val = run_data["M"]
        steps = sorted(run_data["by_step"].keys())
        dmmse_vals = [float(np.mean(run_data["by_step"][s]["dmmse"])) for s in steps]
        ridge_vals = [float(np.mean(run_data["by_step"][s]["ridge"])) for s in steps]
        baseline_vals = [float(np.mean(run_data["by_step"][s]["baseline"])) for s in steps]

        plt.figure(figsize=(8, 4))
        plt.plot(steps, dmmse_vals, label="ED(PT, dMMSE)", marker="o", color="steelblue")
        plt.plot(steps, ridge_vals, label="ED(PT, ridge)", marker="o", color="darkorange")
        plt.plot(steps, baseline_vals, label="ED(dMMSE, ridge)", linestyle="--", color="gray")
        plt.title(f"{run_key} | M = {M_val}")
        plt.xlabel("Training step")
        plt.ylabel("Energy Distance")
        plt.grid(alpha=0.4)
        plt.legend()
        plt.tight_layout()

        run_output_dir = BASE_PLOT_DIR  # Save all plots for this experiment in a single directory
        plot_filename = build_experiment_filename(
            "energy-distance",
            run=run_key,
            tasks=M_val,
            recursion_steps=forward_recursion_steps,
            samples=forward_recursion_samples,
            n_test=N_test,
            prompt_len=PROMPT_LEN,
        )
        plot_path = os.path.join(run_output_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved ED plot: {plot_path}")

        # Per-run CSV of aggregated means
        per_run_csv_filename = build_experiment_filename(
            "energy-distance-summary",
            extension="csv",
            run=run_key,
            tasks=M_val,
            recursion_steps=forward_recursion_steps,
            samples=forward_recursion_samples,
            n_test=N_test,
            prompt_len=PROMPT_LEN,
        )
        per_run_csv_path = os.path.join(run_output_dir, per_run_csv_filename)
        try:
            with open(per_run_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "ed_pt_dmmse_mean", "ed_pt_ridge_mean", "ed_dmmse_ridge_mean"]) 
                for s, a, b, c in zip(steps, dmmse_vals, ridge_vals, baseline_vals):
                    writer.writerow([int(s), float(a), float(b), float(c)])
            print(f"Saved per-run ED CSV: {per_run_csv_path}")
        except Exception as e:
            print(f"Warning: failed to write per-run ED CSV for {run_key}: {e}")

# %%

print(
    f"\nEnergy distance artefacts saved to {SUMMARY_PLOT_DIR} and individual files in {BASE_PLOT_DIR}."
)

# %%
