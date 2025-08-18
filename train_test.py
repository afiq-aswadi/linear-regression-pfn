#%%
from models.model import AutoregressivePFN, bin_y_values, unbin_y_values, construct_sequence
from models.model_config import ModelConfig
from predictive_resampling.predictive_resampling import predictive_resampling_beta_chunked

device = 'mps'


#%%
model_config = ModelConfig(
        d_model=64,
        d_x=8,
        d_y=1,
        n_layers=2,
        n_heads=2,
        d_mlp=4 * 64,
        d_vocab=128,
        n_ctx=32  # 2 * num_examples
    )
    
    # Training hyperparameters
training_config = {
        'device': 'mps',
        'task_size': 8,
        'num_tasks': 1024,
        'noise_var': .25,
        'num_examples': 16,
        'learning_rate': 0.003,
        'training_steps': 2 ** 12,
        'batch_size': 256,
        'eval_batch_size': 1024,
        'print_loss_interval': 100,
        'print_metrics_interval': 1000,
        'n_checkpoints': 10,
    }

#%%

model = AutoregressivePFN(model_config).to(device)
import os
import torch

checkpoint_path = os.path.join('checkpoints', '20250818_182938_model_checkpoint_10.pt')
checkpoint = torch.load(checkpoint_path, map_location='mps')
model.load_state_dict(checkpoint)


#%%
n_points = 10

test_x = torch.randn(n_points,12, model_config.d_x).to(device)
test_y = torch.randn(n_points,12, model_config.d_y).to(device) 

x, y = construct_sequence(test_x, test_y)
print(y)

logits = model(test_x, test_y)[:,-1,:]
print(logits.shape)

# %%
probs = torch.softmax(logits, dim=-1)
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
for i in range(n_points):
    plt.subplot(2, 5, i+1)
    plt.plot(probs[i].detach().cpu().numpy().flatten())
    plt.title(f'Point {i+1} Distribution')
    plt.xlabel('Bin Index')
    plt.ylabel('Probability')
    plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Sample from the probability distribution for each point
sampled_bins = torch.multinomial(probs, num_samples=1)

# Convert bin indices back to continuous values
sampled_y_values = unbin_y_values(sampled_bins, y_min=-3.0, y_max=3.0, n_bins=model_config.d_vocab)

print("Sampled y values:")
for i, y in enumerate(sampled_y_values):
    print(f"Point {i+1}: {y.item():.3f}")
# %%
# %%
print(probs)
# %%
