"""
Plot posterior marginal densities per dimension:
- β_PT samples via predictive resampling KDE
- Gaussian marginals N(μ_r[dim], Σ_r[dim,dim]) from ridge posterior given (X, y)

Saves a D×checkpoints figure per run in PLOTS_DIR.
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
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from predictive_resampling.predictive_resampling import predictive_resampling_beta_chunked

from experiments.experiment_utils import (
    get_device,
    build_checkpoint_path,
    load_model_from_checkpoint,
    build_experiment_filename,
    ensure_experiment_dir,
)
from experiments.experiment_configs import (
    RAVENTOS_SWEEP_MODEL_CONFIG,
    RUNS,
    CHECKPOINTS_DIR,
    PLOTS_DIR,
)


#%%
device = get_device()
model_config = RAVENTOS_SWEEP_MODEL_CONFIG
BASE_PLOT_DIR = ensure_experiment_dir(PLOTS_DIR, __file__)

# Configuration
forward_recursion_steps = 64
forward_recursion_samples = 1000

# Test data (synthetic)
TEST_SEED = 42
rng = np.random.default_rng(TEST_SEED)

N_test = 1
PROMPT_LEN = 8
D = 16
SIGMA2 = 0.25

X_test_all = rng.standard_normal(size=(N_test, PROMPT_LEN, D)).astype(np.float32)
w_true_all = rng.standard_normal(size=(N_test, D)).astype(np.float32)
noise = rng.normal(0, np.sqrt(SIGMA2), size=(N_test, PROMPT_LEN)).astype(np.float32)
y_test_all = np.einsum('tkd,td->tk', X_test_all, w_true_all) + noise

X_test_all = torch.from_numpy(X_test_all).to(device)
y_test_all = torch.from_numpy(y_test_all).unsqueeze(-1).to(device)


#%%

# Use one fixed sequence across checkpoints for comparability
TEST_IDX_TO_PLOT = 0

for run_key, run_info in RUNS.items():
    print(f"\nProcessing {run_key} (task_size={run_info['task_size']})...")
    run_output_dir = BASE_PLOT_DIR  # Save all plots for this experiment in a single directory

    # Sample every 4th checkpoint to reduce load
    selected_checkpoints = run_info['ckpts'][::4]
    print(f"Selected checkpoints: {selected_checkpoints}")

    n_dims = D
    n_checkpoints = len(selected_checkpoints)

    fig, axes = plt.subplots(n_dims, n_checkpoints, figsize=(4*n_checkpoints, 3*n_dims))
    if n_checkpoints == 1:
        axes = axes.reshape(-1, 1)

    # Fixed (X, y)
    x_test = X_test_all[[TEST_IDX_TO_PLOT]]  # (1, K, D)
    y_test = y_test_all[[TEST_IDX_TO_PLOT]]  # (1, K, 1)

    for ckpt_idx, checkpoint_step in enumerate(selected_checkpoints):
        print(f"  Processing checkpoint {checkpoint_step}...")

        # Load model
        checkpoint_path = build_checkpoint_path(
            CHECKPOINTS_DIR, run_info["run_id"], run_info["task_size"], checkpoint_step
        )
        model = load_model_from_checkpoint(model_config, checkpoint_path, device=device)

        # Predictive resampling posterior samples of beta
        beta_pt = predictive_resampling_beta_chunked(
            model,
            model_config,
            forward_recursion_steps=forward_recursion_steps,
            forward_recursion_samples=forward_recursion_samples,
            init_x=x_test,
            init_y=y_test,
        )
        beta_pt = np.asarray(beta_pt)  # (S, D)

        # Ridge posterior parameters: w | (X,y) ~ N(mu_r, Sigma_r)
        XtX = x_test.squeeze(0).T @ x_test.squeeze(0)
        A = XtX + SIGMA2 * torch.eye(D, device=device)
        Sigma_r = torch.linalg.inv(A)
        mu_r = Sigma_r @ x_test.squeeze(0).T @ y_test.squeeze(0)  # (D, 1)

        mu_r_np = mu_r.squeeze(-1).detach().cpu().numpy()  # (D,)
        Sigma_r_np = Sigma_r.detach().cpu().numpy()  # (D, D)

        # Plot per-dimension
        for dim in range(n_dims):
            ax = axes[dim, ckpt_idx]

            beta_values = beta_pt[:, dim]
            if beta_values.ndim > 1:
                beta_values = beta_values.squeeze()

            # Histogram of β_PT samples
            ax.hist(beta_values, bins=50, alpha=0.45, density=True, color='blue', label='β_PT hist')

            # KDE of β_PT samples
            x_range = None
            if len(beta_values) > 1:
                try:
                    kde = gaussian_kde(beta_values)
                    x_min = float(np.min(beta_values))
                    x_max = float(np.max(beta_values))
                    pad = 0.1 * (x_max - x_min + 1e-8)
                    x_range = np.linspace(x_min - pad, x_max + pad, 200)
                    kde_values = kde(x_range)
                    ax.plot(x_range, kde_values, 'g-', lw=2, label='β_PT KDE')
                except Exception:
                    pass

            # Gaussian marginal from ridge posterior
            mu_dim = float(mu_r_np[dim])
            sigma_dim = float(np.sqrt(max(Sigma_r_np[dim, dim], 1e-12)))

            x_gauss_min = mu_dim - 4.0 * sigma_dim
            x_gauss_max = mu_dim + 4.0 * sigma_dim
            if x_range is not None:
                x_lo = min(x_gauss_min, float(np.min(x_range)))
                x_hi = max(x_gauss_max, float(np.max(x_range)))
                x_plot = np.linspace(x_lo, x_hi, 200)
            else:
                x_plot = np.linspace(x_gauss_min, x_gauss_max, 200)

            ax.plot(x_plot, stats.norm.pdf(x_plot, loc=mu_dim, scale=sigma_dim), 'r-', lw=2, label='Ridge marginal')

            # Formatting
            ax.set_xlabel(f'β_{dim+1}')
            ax.set_ylabel('Density')
            ax.set_title(f'Dim {dim+1}, Step {checkpoint_step}')
            ax.grid(True, alpha=0.3)
            if dim == 0 and ckpt_idx == 0:
                ax.legend(fontsize='small')

    fig.suptitle(
        f'Posterior marginals (β_PT KDE vs Ridge N(μ, Σ_ii)) - {run_key} (seq={TEST_IDX_TO_PLOT})',
        fontsize=16,
    )
    plt.tight_layout()

    plot_filename = build_experiment_filename(
        "posterior-marginals",
        run=run_key,
        tasks=run_info["task_size"],
        sequence=TEST_IDX_TO_PLOT,
        ckpt_min=min(selected_checkpoints),
        ckpt_max=max(selected_checkpoints),
        ckpt_count=len(selected_checkpoints),
        recursion_steps=forward_recursion_steps,
        samples=forward_recursion_samples,
    )
    plot_path = os.path.join(run_output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

print(f"\nPosterior marginal analysis complete! Check {BASE_PLOT_DIR} for outputs.")

# %%
