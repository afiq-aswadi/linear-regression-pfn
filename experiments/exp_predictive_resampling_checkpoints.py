#%%

#todo

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from models.model import AutoregressivePFN
from models.model_config import ModelConfig
from predictive_resampling.predictive_resampling_plots import plot_predictive_resampling_from_checkpoints
from samplers.tasks import load_task_distribution_from_pt
from experiments.experiment_utils import (
    get_device,
    get_checkpoints_dir,
    build_checkpoint_path,
    load_model_from_checkpoint,
    load_task_distribution,
    extract_w_pool,
)
from experiments.experiment_configs import (
    RAVENTOS_SWEEP_MODEL_CONFIG
)

device = get_device()
#%%