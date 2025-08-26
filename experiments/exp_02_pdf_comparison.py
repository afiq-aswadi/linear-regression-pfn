"""
Noting the PFNs output a probability distribution over bins, why don't we just compare this to the ridge/dmmse distributions?
"""


#%%
from models.model import AutoregressivePFN, bin_y_values, unbin_y_values, construct_sequence
from models.model_config import ModelConfig
from predictive_resampling.predictive_resampling import predictive_resampling_beta_chunked

# def dmmse(
#     W_pool: torch.Tensor,     # (M, D)
#     X:      torch.Tensor,     # (K, D)
#     y:      torch.Tensor,     # (K,)
#     sigma2: float
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Discrete‐MMSE posterior mean over W_pool using all K pairs.
#     weights_i ∝ exp(-1/(2σ²) * ∑_{j=1}^K (y_j - x_j·w^(i))^2).
#     """
#     # 1) get predictions for each w^(i): shape (K, M)
#     preds = X @ W_pool.T          # (K, M)

#     # 2) residuals per task, then transpose → (M, K)
#     #    y.unsqueeze(1) is (K,1), broadcasts to (K,M)
#     errs = (y.unsqueeze(1) - preds).T  # now (M, K)

#     # 3) unnormalized log‐weights
#     log_w = -0.5 / sigma2 * (errs**2).sum(dim=1)  # (M,)

#     # 4) normalize
#     weights = torch.softmax(log_w, dim=0)         # (M,)
#     # 5) posterior mean
#     mu =  (weights.unsqueeze(1) * W_pool).sum(dim=0)  # (D,)

#     #6) posterior covariance


def ridge(X_test, y_test, sigma2):
    """
    Returns the posterior mean under N(0, I) prior.
    """
    K, D = X_test.shape
    XtX = X_test.T @ X_test                              # (D,D)
    A = XtX + sigma2 * torch.eye(D, device=X_test.device)
    mean= torch.linalg.solve(A, X_test.T @ y_test)      # (D,)
    cov = torch.linalg.inv(A)
    return mean, cov

device = 'cuda'


#%%
model_config = ModelConfig(
        d_model=64,
        d_x=2,
        d_y=1,
        n_layers=2,
        n_heads=2,
        d_mlp=4 * 64,
        d_vocab=64,
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

checkpoint_path = os.path.join('checkpoints', '20250818_124433_model_checkpoint_3.pt')
checkpoint = torch.load(checkpoint_path, map_location='cuda')
model.load_state_dict(checkpoint)




#%%
batch_size = 10
# Generate random weights for linear relationship
w = torch.randn(model_config.d_x, model_config.d_y).to(device)

# Generate random x values
test_x = torch.randn(batch_size, 63, model_config.d_x).to(device)
test_y = test_x @ w 
test_y_original = test_y.clone()

#%%
# Generate y values as linear function of x with noise
noise = torch.randn(batch_size, 63, model_config.d_y).to(device) * 0.5
test_y = test_y_original + noise


x, y = construct_sequence(test_x, test_y)
print(y)

logits = model(test_x, test_y)[:,-2,:]
print(logits.shape)

# %%
probs = torch.softmax(logits, dim=-1)
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
for i in range(batch_size):
    plt.subplot(2, 5, i+1)
    plt.bar(range(len(probs[i])), probs[i].detach().cpu().numpy().flatten())
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