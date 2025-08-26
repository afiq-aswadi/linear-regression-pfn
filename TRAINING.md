# Training Guide

This document explains how to train models in this linear regression PFN codebase.

## Training Scripts

### 1. Basic Training (`train.py`)
Trains a single model with specified parameters.

```bash
python train.py
```

This uses the default configuration in the `if __name__ == "__main__"` block and trains one model.

### 2. Raventos Experiment (`raventos_train.py`)
Runs experiments sweeping over different numbers of pretraining tasks to study memorization vs generalization.

## Raventos Training Arguments

### Core Sweep Parameters
- `--tasks`: List of numbers of pretraining tasks to sweep
  ```bash
  python raventos_train.py --tasks 4 16 64 256 1024
  ```
  Default: `[2^i for i in range(16)]` (1, 2, 4, 8, ..., 65536)

- `--task_size`: List of task dimensionalities 
  ```bash  
  python raventos_train.py --task_size 8 16
  ```
  Default: `[16]`

### Training Configuration
- `--steps`: Training steps per model (default: 500000)
- `--batch`: Batch size (default: 1024) 
- `--eval_batch`: Evaluation batch size (default: 256)
- `--examples`: Number of in-context examples per sequence (default: 64)
- `--noise`: Output noise variance (default: 0.25)
- `--lr`: Maximum learning rate for OneCycleLR scheduler (default: 1e-3)
- `--checkpoints`: Number of model checkpoints to save per run (default: 10)

### Parallel Training (Multi-GPU)
- `--parallel`: Enable multi-GPU parallel training
- `--gpus N`: Use N GPUs (default: all available)

```bash
# Use all available GPUs
python raventos_train.py --parallel --tasks 4 16 64 256

# Use specific number of GPUs
python raventos_train.py --parallel --gpus 4 --tasks 4 16 64 256
```

### Output
- `--out`: Path to save the results plot
  ```bash
  python raventos_train.py --out ./my_results.png
  ```
  Default: `plots/raventos_experiment.png`

## Training Process

### Single Model Training (train.py)
1. **Model Initialization**: Creates AutoregressivePFN with specified config
2. **Data Setup**: 
   - Pretraining distribution (DiscreteTaskDistribution) 
   - True distribution (GaussianTaskDistribution)
3. **Optimization**: Adam optimizer with OneCycleLR scheduler
4. **Training Loop**: 
   - Sample batch from pretraining distribution
   - Convert continuous y-values to discrete bins
   - Forward pass and cross-entropy loss
   - Backward pass and optimization
5. **Checkpointing**: Saves model and task distributions periodically

### Raventos Experiment Training
For each combination of (num_tasks, task_size):
1. **Model Training**: Calls `train()` function from train.py
2. **Evaluation**: 
   - Loads saved task distributions from checkpoints
   - Evaluates model performance on both pretraining and true distributions
   - Compares against baseline methods (dMMSE, Ridge regression)
3. **Metrics Collection**: Records MSE and delta metrics
4. **Plotting**: Creates comprehensive visualization of results

## Parallel Training Logic

When `--parallel` is enabled:

### GPU Assignment
- Uses round-robin assignment: `device_id = task_index % num_gpus`
- Example with 8 tasks, 3 GPUs:
  - GPU 0: trains tasks 0, 3, 6 (sequentially)
  - GPU 1: trains tasks 1, 4, 7 (sequentially) 
  - GPU 2: trains tasks 2, 5 (sequentially)

### Process Management
- Creates multiprocessing pool with `num_gpus` processes
- Each process handles multiple tasks sequentially on its assigned GPU
- Models moved to CPU after training for memory-efficient evaluation

## Model Configuration

The default model architecture in raventos_train.py:
```python
ModelConfig(
    d_model=128,      # Transformer hidden dimension
    d_x=16,           # Input dimension
    d_y=1,            # Output dimension  
    n_layers=2,       # Number of transformer layers
    n_heads=2,        # Number of attention heads
    d_mlp=256,        # MLP hidden dimension (2x d_model)
    d_vocab=128,      # Vocabulary size for y-value binning
    n_ctx=128,        # Context length (2 * num_examples)
    d_head=64,        # Attention head dimension (d_model // n_heads)
    y_min=-7,         # Minimum y-value for binning
    y_max=7,          # Maximum y-value for binning
)
```

## Example Commands

### Quick Test Run
```bash
python raventos_train.py --tasks 4 16 --steps 1000 --batch 256
```

### Full Experiment  
```bash
python raventos_train.py --tasks 4 16 64 256 1024 4096 --steps 500000 --parallel
```

### Custom Configuration
```bash
python raventos_train.py \
  --tasks 8 32 128 512 \
  --task_size 16 \
  --steps 100000 \
  --batch 512 \
  --examples 32 \
  --lr 2e-3 \
  --parallel --gpus 4 \
  --out ./custom_experiment.png
```

## Output Files

### Checkpoints Directory
- Model checkpoints: `{run_id}_model_checkpoint_{i}.pt`
- Task distributions: `{run_id}_pretrain_discrete_{num_tasks}tasks_{task_size}d.pt`
- True distributions: `{run_id}_true_gaussian_{task_size}d.pt`

### Results
- Experiment plot saved to specified `--out` path
- Contains memorization vs generalization analysis across different numbers of pretraining tasks