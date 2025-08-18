# ---------------------------------------------------------------
# Visualize PRIOR Predictive Resampling results
# Runs from saved results.
# Assumes Initial Setup Cell has been run.
# ---------------------------------------------------------------

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from pathlib import Path
from glob import glob
import torch
import os
from typing import Tuple, List
from models.model import AutoregressivePFN
from models.model_config import ModelConfig
from predictive_resampling.predictive_resampling import predictive_resampling_beta_chunked


def plot_prior_predictive_resampling(prior_beta_trajectories: list[np.ndarray], w_pool: np.ndarray, M: int, step_list: list[int], total_steps: int, n_checkpoints: int):
    D = w_pool.shape[1] # Infer dimension D from the loaded W_pool

    all_samples = np.concatenate(prior_beta_trajectories, axis=0)
    x_min = all_samples.min() - 0.5
    x_max = all_samples.max() + 0.5
    x_vals = np.linspace(x_min, x_max, 500) 

    fig, axes = plt.subplots(
        D,
        n_checkpoints,
        figsize=(4 * n_checkpoints, 3 * D),
        sharex="col",
        sharey="row",
    )

    for d in range(D):
        w_vals = w_pool[:, d]
        for c in range(n_checkpoints):
            ax = axes[d, c]
            data = prior_beta_trajectories[c][:, d]
            ax.hist( data, bins=30, density=True, color="steelblue", alpha=0.6, edgecolor="none", label="Transformer" if (d == 0 and c == 0) else None,)
            sns.kdeplot( data, ax=ax, cut=0, color="steelblue", linestyle="--", linewidth=1, label="Transformer KDE" if (d == 0 and c == 0) else None,)
            if M <= 20:
                ymin, ymax = ax.get_ylim()
                y_offset = ymax * 0.02
                ax.plot( w_vals, [y_offset] * len(w_vals), marker='|', linestyle='', color='darkorange', markersize=20, markeredgewidth=2, label='pretraining tasks' if (d == 0 and c == 0) else None)
            else:
                ax.hist( w_vals, bins=min(30, M), density=True, color='darkorange', alpha=0.3, edgecolor='none', label='pretraining tasks' if (d == 0 and c == 0) else None )
            pdf_vals = norm.pdf(x_vals)
            ax.plot( x_vals, pdf_vals, linestyle="--", color="tab:red", label=("N(0,1)" if (d == 0 and c == 0) else None),)
            if c < len(step_list):
                this_step = step_list[c]
                ax.set_title( f"M={M}, step {this_step}/{total_steps},  β_dim {d}" )
            else:
                ax.set_title( f"M={M}, β_dim {d} (Data Missing)" )
    if axes.size > 0:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.show()
    
    # For Jupyter notebooks
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        pass


def load_checkpoints_and_task_distribution(checkpoints_dir: str, run_id: str) -> Tuple[List[str], np.ndarray, int, List[int]]:
    """
    Load model checkpoints and extract w_pool from the corresponding task distribution.
    
    Args:
        checkpoints_dir: Directory containing checkpoints and task distributions
        run_id: The run ID used for saving checkpoints and task distributions
        
    Returns:
        checkpoint_paths: List of checkpoint file paths
        w_pool: Array of shape (M, D) containing the discrete tasks
        M: Number of tasks
        step_list: List of training steps corresponding to each checkpoint
    """
    # Find all checkpoints for this run
    checkpoint_pattern = os.path.join(checkpoints_dir, f"{run_id}_model_checkpoint_*.pt")
    checkpoint_paths = sorted(glob(checkpoint_pattern))
    
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found for run_id {run_id} in {checkpoints_dir}")
    
    # Extract checkpoint indices from checkpoint names
    checkpoint_indices = []
    for path in checkpoint_paths:
        filename = os.path.basename(path)
        # Format: {run_id}_model_checkpoint_{checkpoint_idx}.pt
        checkpoint_idx = int(filename.split('_')[-1].replace('.pt', ''))
        checkpoint_indices.append(checkpoint_idx)
    
    # We need to convert checkpoint indices to actual training steps
    # This requires knowledge of the training configuration
    # For now, we'll return checkpoint indices and let the caller handle the conversion
    step_list = checkpoint_indices
    
    # Load the task distribution to get w_pool
    task_dist_pattern = os.path.join(checkpoints_dir, f"{run_id}_pretrain_discrete_*tasks_*d.pt")
    task_dist_paths = glob(task_dist_pattern)
    
    if not task_dist_paths:
        raise ValueError(f"No task distribution found for run_id {run_id} in {checkpoints_dir}")
    
    task_dist_path = task_dist_paths[0]  # Take the first match
    task_dist_state = torch.load(task_dist_path, map_location='cpu')
    
    if task_dist_state.get('type') != 'discrete':
        raise ValueError(f"Expected discrete task distribution, got {task_dist_state.get('type')}")
    
    w_pool = task_dist_state['tasks'].numpy()  # Shape: (M, D)
    M = w_pool.shape[0]
    
    return checkpoint_paths, w_pool, M, step_list


def run_predictive_resampling_on_checkpoints(
    checkpoint_paths: List[str],
    config: ModelConfig,
    w_pool: np.ndarray,
    forward_recursion_steps: int = 128,
    forward_recursion_samples: int = 1000,
    chunk_size: int = 200,
    device: str = 'cuda'
) -> List[np.ndarray]:
    """
    Run predictive resampling on a list of model checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        config: Model configuration
        w_pool: Array of pretraining tasks
        forward_recursion_steps: Number of steps in forward recursion (T)
        forward_recursion_samples: Number of samples to generate (B_total)
        chunk_size: Batch size for chunked processing
        device: Device to run inference on
        
    Returns:
        List of beta trajectories, one per checkpoint
    """
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    beta_trajectories = []
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Processing checkpoint {i+1}/{len(checkpoint_paths)}: {os.path.basename(checkpoint_path)}")
        
        # Load model
        model = AutoregressivePFN(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        # Run predictive resampling
        with torch.no_grad():
            beta_samples = predictive_resampling_beta_chunked(
                model=model,
                config=config,
                forward_recursion_steps=forward_recursion_steps,
                forward_recursion_samples=forward_recursion_samples,
                chunk_size=chunk_size,
                sample_y=True,
                init_x=None,
                init_y=None
            )
        
        beta_trajectories.append(beta_samples)
        
        # Clear GPU memory
        del model
        del checkpoint
        if device != 'cpu':
            torch.cuda.empty_cache()
    
    return beta_trajectories


def plot_predictive_resampling_from_checkpoints(
    checkpoints_dir: str,
    run_id: str,
    config: ModelConfig,
    training_config: dict = None,
    forward_recursion_steps: int = 128,
    forward_recursion_samples: int = 1000,
    chunk_size: int = 200
):
    """
    Main function to load checkpoints, run predictive resampling, and generate plots.
    
    Args:
        checkpoints_dir: Directory containing checkpoints and task distributions
        run_id: The run ID used for saving checkpoints and task distributions
        config: Model configuration
        training_config: Training configuration dict (optional, used to compute actual step numbers)
        forward_recursion_steps: Number of steps in forward recursion (T)
        forward_recursion_samples: Number of samples to generate (B_total)
        chunk_size: Batch size for chunked processing
    """
    print("Loading checkpoints and task distribution...")
    checkpoint_paths, w_pool, M, step_list = load_checkpoints_and_task_distribution(
        checkpoints_dir, run_id
    )
    
    # Convert checkpoint indices to actual training steps if training_config is provided
    if training_config is not None:
        training_steps = training_config.get('training_steps', 500000)
        n_checkpoints = training_config.get('n_checkpoints', 10)
        if n_checkpoints > 0:
            checkpoint_interval = training_steps // n_checkpoints
            actual_steps = [idx * checkpoint_interval for idx in step_list]
            step_list = actual_steps
            total_steps = training_steps
        else:
            total_steps = max(step_list) if step_list else 1
    else:
        total_steps = max(step_list) if step_list else 1
    
    print(f"Found {len(checkpoint_paths)} checkpoints")
    print(f"Task pool: M={M}, D={w_pool.shape[1]}")
    print(f"Steps: {step_list}")
    
    print("Running predictive resampling on checkpoints...")
    beta_trajectories = run_predictive_resampling_on_checkpoints(
        checkpoint_paths=checkpoint_paths,
        config=config,
        w_pool=w_pool,
        forward_recursion_steps=forward_recursion_steps,
        forward_recursion_samples=forward_recursion_samples,
        chunk_size=chunk_size
    )
    
    print("Generating plots...")
    plot_prior_predictive_resampling(
        prior_beta_trajectories=beta_trajectories,
        w_pool=w_pool,
        M=M,
        step_list=step_list,
        total_steps=total_steps,
        n_checkpoints=len(checkpoint_paths)
    )
    
    return beta_trajectories, w_pool





# print("\n--- Generating Prior Predictive Resampling Plots ---")

# for M in M_list: # Use the M_list defined in the Initial Setup
#     print(f"\nGenerating prior plots for M = {M}")

#     # 1) Load the β̂ trajectories
#     traj_path = PRIOR_SAVE_ROOT / f"M{M}" / "prior_beta_trajectories.pkl"
#     try:
#         with open(traj_path, "rb") as f:
#             prior_beta_trajectories = pickle.load(f)
#     except FileNotFoundError:
#         print(f"Error: Prior trajectory file not found for M={M} at {traj_path}. Skipping.")
#         continue

#     # 2) Load the original W_POOL
#     wpool_path = DRIVE_BASE_DIR / f"W_POOL_M{M}.pkl"
#     try:
#         with open(wpool_path, "rb") as fw:
#             w_pool = pickle.load(fw)  # ndarray of shape (M, D)
#     except FileNotFoundError:
#         print(f"Error: W_POOL file not found for M={M} at {wpool_path}. Skipping prior plot for this M.")
#         continue

#     D = w_pool.shape[1] # Infer dimension D from the loaded W_pool

#     # 3) Re-compute the list of checkpoint paths to get step-numbers
#     ckpt_paths = sorted(glob(str(CKPT_DIR / f"M{M}" / "step_*.pt")))
#     step_list = [ int(Path(p).stem.split("_")[-1]) for p in ckpt_paths ]
#     total_steps = max(step_list)

#     n_checkpoints = len(prior_beta_trajectories)
#     if n_checkpoints == 0:
#         print(f"No prior trajectories loaded for M={M}. Skipping prior plots.")
#         continue

#     if len(step_list) != n_checkpoints:
#           print(f"Warning: Number of checkpoints ({len(step_list)}) does not match number of trajectory files ({n_checkpoints}) for M={M}.")


#     all_samples = np.concatenate(prior_beta_trajectories, axis=0)
#     x_min = all_samples.min() - 0.5
#     x_max = all_samples.max() + 0.5
#     x_vals = np.linspace(x_min, x_max, 500)

#     fig, axes = plt.subplots(
#         D,
#         n_checkpoints,
#         figsize=(4 * n_checkpoints, 3 * D),
#         sharex="col",
#         sharey="row",
#     )
#     axes = np.asarray(axes).reshape(D, n_checkpoints)

#     for d in range(D):
#         w_vals = w_pool[:, d]

#         for c in range(n_checkpoints):
#             ax = axes[d, c]
#             data = prior_beta_trajectories[c][:, d]

#             ax.hist( data, bins=30, density=True, color="steelblue", alpha=0.6, edgecolor="none", label="Transformer" if (d == 0 and c == 0) else None,)
#             sns.kdeplot( data, ax=ax, cut=0, color="steelblue", linestyle="--", linewidth=1, label="Transformer KDE" if (d == 0 and c == 0) else None,)

#             if M <= 20:
#               ymin, ymax = ax.get_ylim()
#               y_offset = ymax * 0.02
#               ax.plot( w_vals, [y_offset] * len(w_vals), marker='|', linestyle='', color='darkorange', markersize=20, markeredgewidth=2, label='pretraining tasks' if (d == 0 and c == 0) else None)
#             else:
#                 ax.hist( w_vals, bins=min(30, M), density=True, color='darkorange', alpha=0.3, edgecolor='none', label='pretraining tasks' if (d == 0 and c == 0) else None )

#             pdf_vals = norm.pdf(x_vals)
#             ax.plot( x_vals, pdf_vals, linestyle="--", color="tab:red", label=("N(0,1)" if (d == 0 and c == 0) else None),)

#             if c < len(step_list):
#                   this_step = step_list[c]
#                   ax.set_title( f"M={M}, step {this_step}/{total_steps},  β_dim {d}" )
#             else:
#                   ax.set_title( f"M={M}, β_dim {d} (Data Missing)" )

#             if d == D - 1:
#                 ax.set_xlabel("β value")
#             if c == 0:
#                 ax.set_ylabel("Density")

#     if axes.size > 0:
#         handles, labels = axes[0, 0].get_legend_handles_labels()
#         fig.legend(handles, labels, loc="upper right")
#     plt.tight_layout()
#     plt.show()


# Example usage
if __name__ == "__main__":
    from models.model_config import ModelConfig
    
    # Example model configuration (should match the one used in training)
    model_config = ModelConfig(
        d_model=64,
        d_x=8,  # Update based on your training config
        d_y=1,
        n_layers=4,
        n_heads=2,
        d_mlp=4 * 64,
        d_vocab=128,
        n_ctx=32
    )
    
    # Example training configuration (should match the one used in training)
    training_config = {
        'training_steps': 500000,  # Update based on your training config
        'n_checkpoints': 10,       # Update based on your training config
        'num_tasks': 1024,         # This is M - number of pretraining tasks
    }
    
    # Example usage - update the run_id based on your actual checkpoints
    checkpoints_dir = "../checkpoints"  # Adjust path as needed
    run_id = "20250818_144926"  # Example run ID - update this
    
    try:
        beta_trajectories, w_pool = plot_predictive_resampling_from_checkpoints(
            checkpoints_dir=checkpoints_dir,
            run_id=run_id,
            config=model_config,
            training_config=training_config,
            forward_recursion_steps=128,
            forward_recursion_samples=1000,
            chunk_size=200
        )
        print(f"Successfully generated plots for {len(beta_trajectories)} checkpoints")
        print(f"W_pool shape: {w_pool.shape}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to update the model_config, training_config, and run_id to match your setup")
