"""
Aims to replicate Figure 2 of Raventos et al.

TODO:
    [] Implement import checkpoints
    [] MAKE clear distinction between task_size and num_tasks
"""

#%%

import torch
import matplotlib.pyplot as plt
from torch.nn import MSELoss


from models.model import AutoregressivePFN
from samplers.tasks import RegressionSequenceDistribution
from baselines import ridge_predictor, dmmse_predictor
from experiments.experiment_utils import (
    get_device,
    build_checkpoint_path,
    get_pretrain_distribution_path,
    get_true_distribution_path,
    load_model_from_checkpoint,
    load_task_distribution,
)
from experiments.experiment_configs import (
    RAVENTOS_SWEEP_MODEL_CONFIG,
    RUNS,
    CHECKPOINTS_DIR,
    PLOTS_DIR
)


device = get_device()
model_config = RAVENTOS_SWEEP_MODEL_CONFIG

#%%
num_examples = 64 #prompt length
batch_size = 256
ckpt_idx = 149999  #what checkpoint we want to use
task_size = 16
NOISE_VARIANCE = 0
MSE_START_POINT = 0 #compute mse from this point onwards

#%%
# m1_run = dict([("m1", RUNS["m1"])])
# m1_run["m1"]["ckpts"] = m1_run["m1"]["ckpts"][-1]


#%%
true_dist = get_true_distribution_path(CHECKPOINTS_DIR, "20250826_162748", task_size) #just hard code for now since common across all.
true_dist_distribution = load_task_distribution(true_dist, device=device)
true_dist_sampler = RegressionSequenceDistribution(true_dist_distribution, noise_variance=NOISE_VARIANCE)

xs_true, ys_true = true_dist_sampler.get_batch(num_examples, batch_size) #we keep this the same for all checkpoints.
xs_true = xs_true.to(device)
ys_true = ys_true.to(device)
ridge_preds_true = ridge_predictor(xs_true, ys_true, 0.25, bound_by_y_min_max=True, y_min=model_config.y_min, y_max=model_config.y_max).to(device) #compute this out of the loop, common for all.
mse_loss = MSELoss()
mse_loss_ridge_true = (1/task_size * mse_loss(ridge_preds_true[:, MSE_START_POINT:,:], ys_true[:, MSE_START_POINT:,:])).item()


#%%
# Store results for plotting
results = {}
for run_key, run_info in RUNS.items():
    num_tasks = run_info['task_size']
    if num_tasks not in results:
        results[num_tasks] = {}

#%%
# Re-run the evaluation loop to collect results
for run_key, run_info in RUNS.items():
    run_id = run_info["run_id"]
    num_tasks = run_info['task_size']
 
    # Build model + load checkpoint
    model_path = build_checkpoint_path(CHECKPOINTS_DIR, run_id, num_tasks, ckpt_idx)
    model = load_model_from_checkpoint(model_config, model_path, device=device)

    # Load matching pretrain task distribution for this run
    print(f"loading model from {model_path}, num_tasks: {num_tasks}, task_size: {task_size}")
    task_distribution_path = get_pretrain_distribution_path(CHECKPOINTS_DIR, run_id, num_tasks, task_size)
    task_distribution = load_task_distribution(task_distribution_path, device=device)

    regression_sequence_distribution = RegressionSequenceDistribution(
        task_distribution, noise_variance=NOISE_VARIANCE
    ).to(device)

    # Pretrain data evaluation
    xs_pretrain, ys_pretrain = regression_sequence_distribution.get_batch(num_examples, batch_size)
    xs_pretrain = xs_pretrain.to(device)
    ys_pretrain = ys_pretrain.to(device)

    dmmse_preds_pretrain = dmmse_predictor(xs_pretrain, ys_pretrain, task_distribution, 0.25, bound_by_y_min_max=True, y_min=model_config.y_min, y_max=model_config.y_max).to(device) # B N 1
    ridge_preds_pretrain = ridge_predictor(xs_pretrain, ys_pretrain, 0.25, bound_by_y_min_max=True, y_min=model_config.y_min, y_max=model_config.y_max).to(device) # B N 1
    probs_pretrain, model_mean_pretrain = model.get_model_mean_prediction(xs_pretrain, ys_pretrain)
    model_mean_pretrain = model_mean_pretrain.unsqueeze(2).to(device)

    # Pretrain MSE losses
    mse_loss_dmmse_pretrain = (1/task_size * mse_loss(dmmse_preds_pretrain[:, MSE_START_POINT:,:], ys_pretrain[:, MSE_START_POINT:,:])).item()
    mse_loss_ridge_pretrain = (1/task_size * mse_loss(ridge_preds_pretrain[:, MSE_START_POINT:,:], ys_pretrain[:, MSE_START_POINT:,:])).item()
    mse_loss_model_pretrain = (1/task_size * mse_loss(model_mean_pretrain[:, MSE_START_POINT:,:], ys_pretrain[:, MSE_START_POINT:,:])).item()
    mse_loss_dmmse_pt_pretrain = (1/task_size * mse_loss(dmmse_preds_pretrain[:, MSE_START_POINT:,:], model_mean_pretrain[:, MSE_START_POINT:,:])).item()
    mse_loss_ridge_pt_pretrain = (1/task_size * mse_loss(ridge_preds_pretrain[:, MSE_START_POINT:,:], model_mean_pretrain[:, MSE_START_POINT:,:])).item()

    print(f"mse_loss_dmmse_pretrain: {mse_loss_dmmse_pretrain}")
    print(f"mse_loss_ridge_pretrain: {mse_loss_ridge_pretrain}")
    print(f"mse_loss_model_pretrain: {mse_loss_model_pretrain}")
    print(f"mse_loss_dmmse_pt_pretrain: {mse_loss_dmmse_pt_pretrain}")
    print(f"mse_loss_ridge_pt_pretrain: {mse_loss_ridge_pt_pretrain}")

    # True data evaluation
    probs_true, model_mean_true = model.get_model_mean_prediction(xs_true, ys_true)
    model_mean_true = model_mean_true.unsqueeze(2).to(device)
    dmmse_preds_true = dmmse_predictor(xs_true, ys_true, task_distribution, 0.25, bound_by_y_min_max=True, y_min=model_config.y_min, y_max=model_config.y_max).to(device)

    # True MSE losses
    mse_loss_dmmse_true = (1/task_size * mse_loss(dmmse_preds_true[:, MSE_START_POINT:,:], ys_true[:, MSE_START_POINT:,:])).item()
    mse_loss_model_true = (1/task_size * mse_loss(model_mean_true[:, MSE_START_POINT:,:], ys_true[:, MSE_START_POINT:,:])).item()
    mse_loss_dmmse_pt_true = (1/task_size * mse_loss(dmmse_preds_true[:, MSE_START_POINT:,:], model_mean_true[:, MSE_START_POINT:,:])).item()
    mse_loss_ridge_pt_true = (1/task_size * mse_loss(ridge_preds_true[:, MSE_START_POINT:,:], model_mean_true[:, MSE_START_POINT:,:])).item()

    # Store results
    results[num_tasks] = {
        'pretrain': {
            'dmmse': mse_loss_dmmse_pretrain,
            'ridge': mse_loss_ridge_pretrain,
            'model': mse_loss_model_pretrain,
            'dmmse_vs_model': mse_loss_dmmse_pt_pretrain,
            'ridge_vs_model': mse_loss_ridge_pt_pretrain
        },
        'true': {
            'dmmse': mse_loss_dmmse_true,
            'ridge': mse_loss_ridge_true,
            'model': mse_loss_model_true,
            'dmmse_vs_model': mse_loss_dmmse_pt_true,
            'ridge_vs_model': mse_loss_ridge_pt_true,
        }
    }

# Create 2x3 subplot layout
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'Prompt Length: {num_examples}, Noise Variance: {NOISE_VARIANCE}, Start Point: {MSE_START_POINT}', fontsize=14)

# Extract sorted num_tasks for x-axis
num_tasks_list = sorted(results.keys())

# Plot 1: Top-left - MSE losses vs num_tasks (pretrain data)
dmmse_pretrain = [results[nt]['pretrain']['dmmse'] for nt in num_tasks_list]
ridge_pretrain = [results[nt]['pretrain']['ridge'] for nt in num_tasks_list]
model_pretrain = [results[nt]['pretrain']['model'] for nt in num_tasks_list]

axes[0,0].semilogx(num_tasks_list, dmmse_pretrain, 'o-', label='DMMSE', color='green', base=2)
axes[0,0].semilogx(num_tasks_list, ridge_pretrain, 's-', label='Ridge', color='blue', base=2)
axes[0,0].semilogx(num_tasks_list, model_pretrain, '^-', label='Model', color='orange', base=2)
axes[0,0].set_xlabel('Number of Tasks')
axes[0,0].set_ylabel('MSE/D')
axes[0,0].set_title('Pretrain Data: MSE vs Ground Truth')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Top-middle - DMMSE vs Model (pretrain data)
dmmse_vs_model_pretrain = [results[nt]['pretrain']['dmmse_vs_model'] for nt in num_tasks_list]

axes[0,1].loglog(num_tasks_list, dmmse_vs_model_pretrain, 'o-', color='blue', label='DMMSE vs Model')
axes[0,1].set_xscale('log', base=2)
axes[0,1].set_xlabel('Number of Tasks')
axes[0,1].set_ylabel('MSE/D')
axes[0,1].set_title('Pretrain Data: DMMSE vs Model Predictions')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Top-right - Ridge vs Model (pretrain data)
ridge_vs_model_pretrain = [results[nt]['pretrain']['ridge_vs_model'] for nt in num_tasks_list]

axes[0,2].loglog(num_tasks_list, ridge_vs_model_pretrain, 's-', color='orange', label='Ridge vs Model')
axes[0,2].set_xscale('log', base=2)
axes[0,2].set_xlabel('Number of Tasks')
axes[0,2].set_ylabel('MSE/D')
axes[0,2].set_title('Pretrain Data: Ridge vs Model Predictions')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# Plot 4: Bottom-left - MSE losses vs num_tasks (true/generalization data)
dmmse_true = [results[nt]['true']['dmmse'] for nt in num_tasks_list]
ridge_true = [results[nt]['true']['ridge'] for nt in num_tasks_list]
model_true = [results[nt]['true']['model'] for nt in num_tasks_list]

axes[1,0].semilogx(num_tasks_list, dmmse_true, 'o-', label='DMMSE', color='green', base=2)
axes[1,0].semilogx(num_tasks_list, ridge_true, 's-', label='Ridge', color='blue', base=2)
axes[1,0].semilogx(num_tasks_list, model_true, '^-', label='Model', color='orange', base=2)
axes[1,0].set_xlabel('Number of Tasks')
axes[1,0].set_ylabel('MSE/D')
axes[1,0].set_title('Generalization Data: MSE vs Ground Truth')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 5: Bottom-middle - DMMSE vs Model (true/generalization data)
dmmse_vs_model_true = [results[nt]['true']['dmmse_vs_model'] for nt in num_tasks_list]

axes[1,1].loglog(num_tasks_list, dmmse_vs_model_true, 'o-', color='blue', label='DMMSE vs Model')
axes[1,1].set_xscale('log', base=2)
axes[1,1].set_xlabel('Number of Tasks')
axes[1,1].set_ylabel('MSE/D')
axes[1,1].set_title('Generalization Data: DMMSE vs Model Predictions')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# Plot 6: Bottom-right - Ridge vs Model (true/generalization data)
ridge_vs_model_true = [results[nt]['true']['ridge_vs_model'] for nt in num_tasks_list]

axes[1,2].loglog(num_tasks_list, ridge_vs_model_true, 's-', color='orange', label='Ridge vs Model')
axes[1,2].set_xscale('log', base=2)
axes[1,2].set_xlabel('Number of Tasks')
axes[1,2].set_ylabel('MSE/D')
axes[1,2].set_title('Generalization Data: Ridge vs Model Predictions')
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/raventos_replication_task_size_{task_size}_prompt_len_{num_examples}_start_point_{MSE_START_POINT}.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
