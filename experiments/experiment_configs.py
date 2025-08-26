"""
Define model configs, checkpoint ids here
"""
import os

from models.model_config import ModelConfig
from train import train_logarithmic_checkpoints

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


RAVENTOS_SWEEP_MODEL_CONFIG = ModelConfig(
    d_model=128,
    d_x=16,
    d_y=1,
    n_layers=2,
    n_heads=2,
    d_mlp=256,
    d_vocab=128,
    n_ctx=128,
    d_head = 64,
    y_min=-7,
    y_max=7,
)

training_config = {
    "training_steps": 150000,
    "n_checkpoints": 10,
}

checkpoints = train_logarithmic_checkpoints(training_config["training_steps"], training_config["n_checkpoints"])

RUNS = {
    "m1": {"run_id": "20250818_143023", "task_size": 1, "ckpts": checkpoints},
    "m2": {"run_id": "20250818_170107", "task_size": 16, "ckpts": checkpoints},
    "m3": {"run_id": "20250818_194416", "task_size": 256, "ckpts": checkpoints},
    "m4": {"run_id": "20250818_222551", "task_size": 4096, "ckpts": checkpoints},
    "m5": {"run_id": "20250819_010712", "task_size": 65536, "ckpts": checkpoints},
    "m6": {"run_id": "20250819_034904", "task_size": 1048576, "ckpts": checkpoints},
    "m7": {"run_id": "20250826_011959", "task_size": 2, "ckpts": checkpoints},
}


if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"PLOTS_DIR: {PLOTS_DIR}")
    print(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")