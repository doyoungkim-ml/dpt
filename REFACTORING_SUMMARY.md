# Code Refactoring Summary

## Overview
The codebase has been refactored to create a clean structure with:
- `train_online_unified.py` for online training (no more `train_online.py`)
- `eval_online.py` for online evaluation
- `train.py` for offline training
- `eval.py` for offline evaluation
- 4 standardized sbatch/bash file pairs for each training/evaluation scenario
- YAML config file support for all scripts
- Organized model saving by job ID or timestamp

## New File Structure

### Training Scripts
- `train_online_unified.py` - Online training with confidence functions
- `train.py` - Offline training from datasets

### Evaluation Scripts
- `eval_online.py` - Online evaluation with generated environments
- `eval.py` - Offline evaluation with datasets

### Batch/Shell Scripts
- `train_online.sbatch` / `train_online.sh` - Online training via SLURM/bash
- `train_offline.sbatch` / `train_offline.sh` - Offline training via SLURM/bash
- `eval_online.sbatch` / `eval_online.sh` - Online evaluation via SLURM/bash
- `eval_offline.sbatch` / `eval_offline.sh` - Offline evaluation via SLURM/bash

### Configuration
- `configs/` directory with example YAML configuration files
- `configs/bandit_offline.yaml` - Offline bandit training config
- `configs/bandit_online.yaml` - Online bandit training config
- `configs/linear_bandit_offline.yaml` - Linear bandit offline training config
- `configs/bandit_eval.yaml` - Evaluation config

## Model Saving Structure

Models are now saved in organized directories:
```
models/
  ├── bandit/
  │   ├── 123/                    # Job ID or timestamp
  │   │   ├── config.yaml         # Training configuration
  │   │   ├── logs.txt           # Training logs
  │   │   ├── epoch1.pt          # Model checkpoints
  │   │   ├── epoch10.pt
  │   │   ├── final_model.pt     # Final trained model
  │   │   └── train_loss.png     # Loss plots
  │   └── 124/
  └── linear_bandit/
      └── 125/
```

## Usage Examples

### Training
```bash
# Offline training with SLURM
sbatch train_offline.sbatch configs/bandit_offline.yaml

# Online training with bash
./train_online.sh configs/bandit_online.yaml
```

### Evaluation
```bash
# Offline evaluation
./eval_offline.sh configs/bandit_eval.yaml models/bandit/123/epoch100.pt

# Online evaluation
sbatch eval_online.sbatch configs/bandit_eval.yaml models/bandit/123/final_model.pt
```

## Features

### All Scripts Support:
- YAML configuration files (`--config`)
- Direct model path specification (`--model_path`)
- Job ID or timestamp-based experiment organization
- Automatic directory creation and config saving

### Confidence Functions (Online Training):
- Linear confidence scheduling
- Exponential confidence scheduling
- Stepped confidence scheduling
- Constant confidence values

## Cleanup
Old script files have been moved to `old_scripts/` directory for reference.