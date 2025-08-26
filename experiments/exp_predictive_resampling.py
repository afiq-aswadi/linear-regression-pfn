"""
This script tests the unconditional predictive resampling method on a trained transformer. Make sure the config fits!
"""


#%%
from models.model import AutoregressivePFN, bin_y_values, unbin_y_values, construct_sequence
from models.model_config import ModelConfig
from predictive_resampling.predictive_resampling import predictive_resampling_beta_chunked, predictive_resampling_beta
from experiments.experiment_utils import (
    get_device,
    get_checkpoints_dir,
    load_model_from_checkpoint,
)
from experiments.experiment_configs import (
    RAVENTOS_SWEEP_MODEL_CONFIG,
)
device = get_device()



#%%
model_config = RAVENTOS_SWEEP_MODEL_CONFIG

#%%

model = None
import os
import torch

# insert checkpoint path here
CHECKPOINTS_DIR = get_checkpoints_dir()
checkpoint_path = os.path.join(CHECKPOINTS_DIR, '20250818_124433_model_checkpoint_3.pt')
model = load_model_from_checkpoint(model_config, checkpoint_path, device=device)

#%%

beta_hat, y = predictive_resampling_beta_chunked(model, model_config, forward_recursion_steps=64, forward_recursion_samples=10000, save_y=True)

# %%
print(beta_hat.shape)
print(y.shape)

# %%
print(y[0])
# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot histogram and overlay normal for β₁
counts1, bins1, _ = ax1.hist(beta_hat[:, 0], bins=50, alpha=0.7, density=True)
x1 = np.linspace(min(bins1), max(bins1), 100)
ax1.plot(x1, norm.pdf(x1, 0, 1), 'r-', lw=2, label='N(0,1)')
ax1.set_xlabel('β₁')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of β₁')
ax1.grid(True)
ax1.legend()

# Plot histogram and overlay normal for β₂
counts2, bins2, _ = ax2.hist(beta_hat[:, 1], bins=50, alpha=0.7, density=True)
x2 = np.linspace(min(bins2), max(bins2), 100)
ax2.plot(x2, norm.pdf(x2, 0, 1), 'r-', lw=2, label='N(0,1)')
ax2.set_xlabel('β₂')
ax2.set_ylabel('Density')
ax2.set_title('Distribution of β₂')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# %%
