"""
training the transformer on synthetic in-context regression task
"""

#%%

import torch
from tqdm import tqdm
import os
from datetime import datetime

from evals import ICLEvaluator
from models.model import AutoregressivePFN, bin_y_values, unbin_y_values
from models.model_config import ModelConfig
from samplers.tasks import RegressionSequenceDistribution
from samplers.tasks import DiscreteTaskDistribution, GaussianTaskDistribution

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
#%%

import numpy as np
def train_logarithmic_checkpoints(training_steps:int, n_checkpoints:int):
    """
    Build checkpoint indices C = C_linear ∪ C_log per appendix A.4.

    - C_linear = {0, 100, 200, ..., T}  (mapped to 0-based steps, so T -> T-1)
    - C_log    = { floor(T^(j/(N-1))) : j = 0, 1, ..., N-1 }  (also mapped to ≤ T-1)

    Returns a set of step indices in [0, training_steps-1].
    """
    T = int(training_steps)
    N = int(max(1, n_checkpoints))

    # Linear checkpoints every 100 steps, include 0 and T (map T to T-1 for 0-based loop)
    linear_raw = np.arange(0, T + 1, 10000, dtype=int)
    linear_mapped = {min(int(s), T - 1) for s in linear_raw} if T > 0 else {0}

    # Logarithmically spaced checkpoints: floor(T^(j/(N-1))) for j in [0, N-1]
    if N == 1:
        log_raw = np.array([T], dtype=int)
    else:
        # Using logspace to implement T^(j/(N-1)) = 10^{log10(T) * j/(N-1)}
        log_raw = np.floor(np.logspace(0, np.log10(T), num=N)).astype(int)
    log_mapped = {min(int(s), T - 1) for s in log_raw} if T > 0 else {0}

    checkpoint_steps_set = linear_mapped.union(log_mapped)
    checkpoint_steps_set = sorted(list(checkpoint_steps_set))
    return checkpoint_steps_set

#%%
def train(config: ModelConfig, training_config: dict, print_model_dimensionality: bool = False):
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.

    If logarithmic_checkpoints is True, the checkpoints will be saved at logarithmic intervals for the first 20% of training.
    """

    # model initialisation
    print("initialising model")
    # Determine device for this training run (allows per-process GPU assignment)
    run_device = training_config.get('device', device)
    if run_device.startswith('cuda') and not torch.cuda.is_available():
        run_device = 'cpu'
    model = AutoregressivePFN(config).to(run_device)    
    # model = torch.compile(model)
    model.train()

    if print_model_dimensionality:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params:,}")

    # initialise 'pretraining' data source (for training on fixed task set)
    print("initialising data (pretrain)")
    pretrain_dist = RegressionSequenceDistribution(
        task_distribution=DiscreteTaskDistribution(
            num_tasks=training_config['num_tasks'],
            task_size=training_config['task_size'],
        ),
        noise_variance=training_config['noise_var'],
    ).to(run_device)

    # initialise 'true' data source (for evaluation, including unseen tasks)
    print("initialising data (true)")
    true_dist = RegressionSequenceDistribution(
        task_distribution=GaussianTaskDistribution(
            task_size=training_config['task_size'],
        ),
        noise_variance=training_config['noise_var'],
    ).to(run_device)

    # save task distributions for reproducibility/inspection
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    pretrain_td_path = os.path.join(
        checkpoints_dir,
        f"{run_id}_pretrain_discrete_{training_config['num_tasks']}tasks_{training_config['task_size']}d.pt",
    )
    true_td_path = os.path.join(
        checkpoints_dir,
        f"{run_id}_true_gaussian_{training_config['task_size']}d.pt",
    )
    pretrain_dist.task_distribution.save(pretrain_td_path)
    true_dist.task_distribution.save(true_td_path)

    # initialise evaluations
    print("initialising evaluator")
    # evaluator = ICLEvaluator(
    #     pretrain_dist=pretrain_dist,
    #     true_dist=true_dist,
    #     max_examples=training_config['num_examples'],
    #     eval_batch_size=training_config['eval_batch_size'],
    # )
    # initialise torch optimiser
    print("initialising optimiser and scheduler")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'], # unused, overwritten by scheduler
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=training_config['learning_rate'],
        anneal_strategy='linear',
        total_steps=training_config['training_steps'],
        pct_start=0.50,
        div_factor=training_config['training_steps'] / 2 - 1,
        final_div_factor=training_config['training_steps'] / 2 - 1,
        cycle_momentum=False, # N/A, but required to avoid error
    )

    # training loop
    print("starting training loop")
    for step in tqdm(range(training_config['training_steps']), desc="training..."):
        # training step
        xs, ys = pretrain_dist.get_batch(
            num_examples=training_config['num_examples'],
            batch_size=training_config['batch_size'],
        )
        
        # Convert continuous y values to discrete bin indices for training
        y_bins = bin_y_values(ys, y_min=config.y_min, y_max=config.y_max, n_bins=config.d_vocab)  # [batch, num_examples] 
        
        logits = model(xs, ys) # note: forward loop uses ys, but we train on y_bins
        # Only compute loss on even indices (where we predict y)
        # logits shape: [batch, 2*num_examples, d_vocab]
        # y_bins shape: [batch, num_examples]
        y_pred_logits = logits[:, 0::2]  # predictions at even indices [batch, num_examples, d_vocab]
        # Reshape for cross-entropy loss
        y_pred_flat = y_pred_logits.view(-1, config.d_vocab)  # [batch*num_examples, d_vocab]
        y_bins_flat = y_bins.view(-1)  # [batch*num_examples]
        
        loss = torch.nn.functional.cross_entropy(y_pred_flat, y_bins_flat)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # log some metrics to stdout
        if step % training_config['print_loss_interval'] == 0:
            tqdm.write(f"step {step} loss:")
            tqdm.write(f"  {'batch/loss':<30}: {loss.item():.2f}")

        if training_config.get('n_checkpoints') is not None:
            if training_config.get('logarithmic_checkpoints', False):
                if step == 0:  # Compute checkpoint steps once at the beginning
                    checkpoint_steps_set = train_logarithmic_checkpoints(training_config['training_steps'], training_config['n_checkpoints'])
                    globals()['checkpoint_steps_set'] = checkpoint_steps_set  # Store for reuse
                
                if step in globals().get('checkpoint_steps_set', set()):
                    model.save(os.path.join(checkpoints_dir, f"{run_id}_model_{training_config['num_tasks']}tasks_step_{step}.pt"))
            else:
                checkpoint_interval = training_config['training_steps'] // training_config['n_checkpoints']
                if step % checkpoint_interval == 0:
                    model.save(os.path.join(checkpoints_dir, f"{run_id}_model_{training_config['num_tasks']}tasks_step_{step}.pt"))

    return run_id, model, checkpoints_dir

#%%
if __name__ == "__main__":
    # Model architecture config
    model_config = ModelConfig(
        d_model=64,
        d_x=2,
        d_y=1,
        n_layers=2,
        n_heads=2,
        d_mlp=4 * 64,
        d_vocab=64,
        n_ctx=128,  # 2 * num_examples
        y_min=-6.0,
        y_max=6.0,
    )
    
    # Training hyperparameters
    training_config = {
        'device': 'cuda',
        'task_size': 2,
        'num_tasks': 2,
        'noise_var': .25,
        'num_examples': 64,
        'learning_rate': 0.003,
        'training_steps': 1000,
        'batch_size': 1024,
        'eval_batch_size': 1024,
        'print_loss_interval': 100,
        'print_metrics_interval': 1000,
        'n_checkpoints': 10,
        'logarithmic_checkpoints': True,
    }
    
    run_id, model, checkpoints_dir = train(model_config, training_config, print_model_dimensionality=True)


# %%
