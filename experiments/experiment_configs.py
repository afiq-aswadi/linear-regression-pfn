"""
Define model configs, checkpoint ids here
"""
import os

from models.model_config import ModelConfig

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

RAVENTOS_SWEEP_MODEL_CONFIG = ModelConfig(
    d_model=64,
    d_x=2,
    d_y=1,
    n_layers=2,
    n_heads=2,
    d_mlp=4 * 64,
    d_vocab=64,
    n_ctx=128,
    y_min=-6.0,
    y_max=6.0,
)

RUNS = {
    "m1": {"run_id": "20250818_143023", "task_size": 1, "ckpts": [0, 1, 2, 3, 4]},
    "m2": {"run_id": "20250818_170107", "task_size": 16, "ckpts": [0, 1, 2, 3, 4]},
    "m3": {"run_id": "20250818_194416", "task_size": 256, "ckpts": [0, 1, 2, 3, 4]},
    "m4": {"run_id": "20250818_222551", "task_size": 4096, "ckpts": [0, 1, 2, 3, 4]},
    "m5": {"run_id": "20250819_010712", "task_size": 65536, "ckpts": [0, 1, 2, 3, 4]},
    "m6": {"run_id": "20250819_034904", "task_size": 1048576, "ckpts": [0, 1, 2, 3, 4]},
    "m7": {"run_id": "20250826_011959", "task_size": 2, "ckpts": [0, 1, 2, 3, 4]},
}


if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"PLOTS_DIR: {PLOTS_DIR}")
    print(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")