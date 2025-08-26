# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based implementation of a Prior-Fitted Network (PFN) for in-context linear regression, following transformer architecture principles. The model learns to perform regression on synthetic tasks during training and can generalize to new regression tasks at inference time.

## Key Architecture Components

- **AutoregressivePFN**: Main transformer model that processes (x,y) pairs in an autoregressive sequence format
- **Task Distributions**: Two types of synthetic task generators:
  - `DiscreteTaskDistribution`: Fixed set of linear regression tasks for pretraining
  - `GaussianTaskDistribution`: Gaussian-sampled tasks for evaluation
- **Predictive Resampling**: Implementation for analyzing model predictions and task learning
- **Baselines**: DMMSE and Ridge regression implementations for comparison

## Directory Structure

- `models/`: Core model implementations and configuration
- `samplers/`: Task distribution generators and sequence samplers
- `experiments/`: Various experimental scripts and analysis
- `predictive_resampling/`: Predictive resampling implementation and analysis
- `checkpoints/`: Saved model states and task distributions

## Common Commands

### Training
```bash
# Basic training with default parameters
python train.py

# Raventos-style experiment (task diversity sweep)
python raventos_train.py

# Custom training parameters
python raventos_train.py --steps 10000 --batch 512 --tasks 4 16 64 256 1024
```

### Evaluation
```bash
# Run evaluations on trained models
python evals.py
```

### Experiments
```bash
# Various experimental analysis scripts
python experiments/exp_00_predictive_resampling.py
python experiments/exp_01_raventos_plots.py
python experiments/exp_02_pdf_comparison.py
```

### Testing
```bash
# Run existing tests
python predictive_resampling/predictive_resampling_test.py
```

## Model Configuration

Model parameters are defined in `models/model_config.py` with the `ModelConfig` dataclass:
- `d_model`: 64 (transformer dimension)
- `d_x`: 8 (input dimension)
- `d_y`: 1 (output dimension)
- `d_vocab`: 128 (vocabulary size for y-value binning)
- `n_ctx`: 100 (context length)
- `n_layers`: 4, `n_heads`: 2

## Key Implementation Details

- **Sequence Construction**: Input pairs are arranged as autoregressive sequences: (x₁,0), (x₁,y₁), (x₂,0), (x₂,y₂)...
- **Y-value Binning**: Continuous y-values are discretized into bins for transformer vocabulary
- **Device Support**: Automatic detection of CUDA, MPS, or CPU
- **Checkpointing**: Models and task distributions are automatically saved with timestamps

## Dependencies

Uses `uv` for dependency management. Key dependencies include:
- PyTorch 2.8.0+
- JAX for typing annotations
- einops, matplotlib, seaborn, tqdm
- Requires Python 3.13+