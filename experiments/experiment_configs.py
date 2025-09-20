"""
Define model configs, checkpoint ids here
"""
#%%
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from models.config import ModelConfig, TrainConfig
from train import train_logarithmic_checkpoints


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

training_config = TrainConfig(training_steps=150000, n_checkpoints=10)

checkpoints = train_logarithmic_checkpoints(training_config.training_steps, training_config.n_checkpoints)


# hardcoding checkpoints for now
RUNS = { 
    "m1": {"run_id": "20250826_110937", "task_size": 1, "ckpts": checkpoints},
    "m2": {"run_id": "20250826_110937", "task_size": 2, "ckpts": checkpoints},
    "m3": {"run_id": "20250826_110937", "task_size": 4, "ckpts": checkpoints},
    "m4": {"run_id": "20250826_110937", "task_size": 8, "ckpts": checkpoints},
    "m5": {"run_id": "20250826_125355", "task_size": 16, "ckpts": checkpoints},
    "m6": {"run_id": "20250826_125525", "task_size": 32, "ckpts": checkpoints},
    "m7": {"run_id": "20250826_125534", "task_size": 64, "ckpts": checkpoints},
    "m8": {"run_id": "20250826_125538", "task_size": 128, "ckpts": checkpoints},
    "m9": {"run_id": "20250826_143957", "task_size": 256, "ckpts": checkpoints},
    "m10": {"run_id": "20250826_143959", "task_size": 512, "ckpts": checkpoints},
    "m11": {"run_id": "20250826_144140", "task_size": 1024, "ckpts": checkpoints},
    "m12": {"run_id": "20250826_144306", "task_size": 2048, "ckpts": checkpoints},
    "m13": {"run_id": "20250826_162456", "task_size": 4096, "ckpts": checkpoints},
    "m14": {"run_id": "20250826_162743", "task_size": 8192, "ckpts": checkpoints},
    "m15": {"run_id": "20250826_162744", "task_size": 16384, "ckpts": checkpoints},
    "m16": {"run_id": "20250826_162748", "task_size": 32768, "ckpts": checkpoints},
}


if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(checkpoints)
    print(f"PLOTS_DIR: {PLOTS_DIR}")
    print(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")
# %%
