"""
Shared utilities for experiment scripts:
- Device selection
- Default model config factory
- Checkpoint directory and path helpers
- Model loading from checkpoints
- Task distribution loading helpers
"""

import os
from typing import Optional

import numpy as np
import torch

from models.model import AutoregressivePFN
from models.model_config import ModelConfig
from samplers.tasks import load_task_distribution_from_pt as _load_td


def get_device(prefer: str = "cuda") -> str:
    """
    Return the best available device string: "cuda" > "mps" > "cpu".
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_checkpoints_dir() -> str:
    """
    Resolve checkpoints directory, preferring CHECKPOINTS_DIR env var, otherwise
    experiments/checkpoints relative to this file.
    """
    env = os.environ.get("CHECKPOINTS_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(__file__), "checkpoints")


def build_checkpoint_path(checkpoints_dir: str, run_id: str, ckpt_idx: int) -> str:
    return os.path.join(checkpoints_dir, f"{run_id}_model_step_{ckpt_idx}.pt") #todo: make this better


def get_pretrain_distribution_path(checkpoints_dir: str, run_id: str, num_tasks: int, task_size: int) -> str:
    return os.path.join(checkpoints_dir, f"{run_id}_pretrain_discrete_{num_tasks}tasks_{task_size}d.pt")


def get_true_distribution_path(checkpoints_dir: str, run_id: str, task_size: int) -> str:
    return os.path.join(checkpoints_dir, f"{run_id}_true_gaussian_{task_size}d.pt")


def load_model_from_checkpoint(config: ModelConfig, checkpoint_path: str, device: Optional[str] = None) -> AutoregressivePFN:
    """
    Create an AutoregressivePFN with the provided config and load weights from checkpoint.
    """
    if device is None:
        device = get_device()
    model = AutoregressivePFN(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model = torch.compile(model)
    return model


def load_task_distribution(file_path: str, device: Optional[str] = None):
    if device is None:
        device = get_device()
    return _load_td(file_path, device=device)


def extract_w_pool(task_distribution) -> Optional[np.ndarray]:
    """
    Extract task pool (weights) as a numpy array if present in a loaded distribution.
    Returns None if unavailable.
    """
    w_pool = None
    if isinstance(task_distribution, dict) and "tasks" in task_distribution:
        w_pool = task_distribution["tasks"]
    elif hasattr(task_distribution, "tasks"):
        w_pool = getattr(task_distribution, "tasks")
    else:
        # Assume the object itself may be an array-like pool
        w_pool = task_distribution

    if isinstance(w_pool, torch.Tensor):
        return w_pool.detach().cpu().numpy()
    if isinstance(w_pool, np.ndarray):
        return w_pool
    return None


__all__ = [
    "get_device",
    "default_model_config",
    "get_checkpoints_dir",
    "build_checkpoint_path",
    "get_pretrain_distribution_path",
    "get_true_distribution_path",
    "load_model_from_checkpoint",
    "load_task_distribution",
    "extract_w_pool",
]
