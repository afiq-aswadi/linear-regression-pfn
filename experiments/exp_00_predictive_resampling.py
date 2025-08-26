"""
This script tests the unconditional predictive resampling method on a trained transformer. Make sure the config fits!
"""


#%%
from models.model import AutoregressivePFN, bin_y_values, unbin_y_values, construct_sequence
from models.model_config import ModelConfig
from predictive_resampling.predictive_resampling import predictive_resampling_beta_chunked, predictive_resampling_beta

device = 'cuda'


#%%
model_config = ModelConfig(
        d_model=64,
        d_x=2,
        d_y=1,
        n_layers=2,
        n_heads=2,
        d_mlp=4 * 64,
        d_vocab=64
        n_ctx=128  # 2 * num_examples
    )
    
    # Training hyperparameters
training_config = {
        'device': 'cuda',
        'task_size': 2,
        'num_tasks': 2 ** 16,
        'noise_var': .25,
        'num_examples': 64,
        'learning_rate': 0.003,
        'training_steps': 100000,
        'batch_size': 256,
        'eval_batch_size': 1024,
        'print_loss_interval': 100,
        'print_metrics_interval': 1000,
        'n_checkpoints': 3,
    }

#%%

model = AutoregressivePFN(model_config).to(device)
import os
import torch

# insert checkpoint path here
checkpoint_path = os.path.join('checkpoints', '20250818_124433_model_checkpoint_3.pt')
checkpoint = torch.load(checkpoint_path, map_location='cuda')
model.load_state_dict(checkpoint)

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
