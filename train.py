"""
training the transformer on synthetic in-context regression task
"""
import dotenv; dotenv.load_dotenv() # environment before imports

import torch
import tqdm

from evals import ICLEvaluator
from model import InContextRegressionTransformer
from tasks import RegressionSequenceDistribution
from tasks import DiscreteTaskDistribution, GaussianTaskDistribution



def train(config):
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
    # special code if device is 'xla'
    XLA = (config['device'] == 'xla')
    if XLA:
        print("device is 'xla'! some special code will run...")
        print("importing torch_xla...")
        import torch_xla.core.xla_model as xm
        print("configuring default XLA device...")
        device = xm.xla_device()
        print("xla ready!")
    else:
        device = config['device']

    # model initialisation
    print("initialising model")
    model = InContextRegressionTransformer(
        task_size=config['task_size'],
        max_examples=config['num_examples'],
        embed_size=config['embed_size'],
        mlp_size=config['embed_size'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
    ).to(device)
    model.train()

    # initialise 'pretraining' data source (for training on fixed task set)
    print("initialising data (pretrain)")
    pretrain_dist = RegressionSequenceDistribution(
        task_distribution=DiscreteTaskDistribution(
            num_tasks=config['num_tasks'],
            task_size=config['task_size'],
        ),
        noise_variance=config['noise_var'],
    ).to(device)

    # initialise 'true' data source (for evaluation, including unseen tasks)
    print("initialising data (true)")
    true_dist = RegressionSequenceDistribution(
        task_distribution=GaussianTaskDistribution(
            task_size=config['task_size'],
        ),
        noise_variance=config['noise_var'],
    ).to(device)


    # initialise evaluations
    print("initialising evaluator")
    if XLA: xm.mark_step()
    evaluator = ICLEvaluator(
        pretrain_dist=pretrain_dist,
        true_dist=true_dist,
        max_examples=config['num_examples'],
        eval_batch_size=config['eval_batch_size'],
    )
    if XLA: xm.mark_step()

    # initialise torch optimiser
    print("initialising optimiser and scheduler")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'], # unused, overwritten by scheduler
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config['learning_rate'],
        anneal_strategy='linear',
        total_steps=config['training_steps'],
        pct_start=0.50,
        div_factor=config['training_steps'] / 2 - 1,
        final_div_factor=config['training_steps'] / 2 - 1,
        cycle_momentum=False, # N/A, but required to avoid error
    )

    # training loop
    print("starting training loop")
    if XLA: print("note: first two iterations slow while XLA compiles")
    for step in tqdm.trange(config['training_steps'], desc="training..."):
        # training step
        if XLA: xm.mark_step()
        xs, ys = pretrain_dist.get_batch(
            num_examples=config['num_examples'],
            batch_size=config['batch_size'],
        )
        ys_pred = model(xs, ys)
        loss = (ys - ys_pred).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if XLA: xm.mark_step()

        # log some metrics to stdout
        if step % config['print_loss_interval'] == 0:
            tqdm.tqdm.write(f"step {step} loss:")
            tqdm.tqdm.write(f"  {'batch/loss':<30}: {loss.item():.2f}")
        if step % config['print_metrics_interval'] == 0:
            if XLA: xm.mark_step()
            model.eval()
            metrics = evaluator(model)
            model.train()
            if XLA: xm.mark_step()
            tqdm.tqdm.write(f"step {step} metrics:")
            for metric, value in metrics.items():
                tqdm.tqdm.write(f"  {metric:<30}: {value:.2f}")

    return model


if __name__ == "__main__":
    config = {
        'device': 'xla',
        'task_size': 8,
        'num_tasks': 8,
        'noise_var': .25,
        'num_examples': 16,
        'embed_size': 64,
        'num_heads': 4,
        'num_layers': 2,
        'learning_rate': 0.003,
        'training_steps': 1024,
        'batch_size': 256,
        'eval_batch_size': 1024,
        'print_loss_interval': 100,
        'print_metrics_interval': 1000,
    }
    _model = train(config)

