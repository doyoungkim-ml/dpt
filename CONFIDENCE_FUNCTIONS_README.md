# Unified Confidence Function System

This document describes the unified confidence function system implemented in `train_online_unified.py` and `eval_online.py`.

## Overview

The system now supports multiple confidence functions through a single training script, replacing the previous separate scripts:
- `train_online_confidence_deprecated.py` (previously `train_online_confidence.py`)
- `train_online_confidence_exp_deprecated.py` (previously `train_online_confidence_exp.py`)

## Confidence Functions

### 1. Linear Confidence
- **Type**: `linear`
- **Formula**: `confidence = confidence_start * (1 - t/H) + 1.0 * (t/H)`
- **Parameters**: `--confidence_start` (default: 0.3)
- **Description**: Linearly interpolates from `confidence_start` to 1.0 over the horizon

### 2. Exponential Confidence
- **Type**: `exponential`
- **Formula**: `confidence = 1.0 - exp(-t/lambda)`
- **Parameters**: `--confidence_lambda` (default: 40)
- **Description**: Exponentially approaches 1.0 with rate parameter lambda

### 3. Stepped Confidence (New)
- **Type**: `stepped`
- **Formula**: 
  - `confidence = confidence_start + (1.0 - confidence_start) * (t/max_position)` for `t < max_position`
  - `confidence = 1.0` for `t >= max_position`
- **Parameters**: `--confidence_start` (default: 0.3), `--max_position` (default: horizon/2)
- **Description**: Linearly increases to 1.0 until `max_position`, then remains at 1.0

### 4. Constant Confidence
- **Type**: `constant`
- **Formula**: `confidence = confidence_value`
- **Parameters**: `--confidence_value` (default: 1.0)
- **Description**: Uses a constant confidence value throughout training

## Usage

### Training
```bash
# Linear confidence
python3 train_online_unified.py --env bandit --confidence_type linear --confidence_start 0.3

# Exponential confidence
python3 train_online_unified.py --env bandit --confidence_type exponential --confidence_lambda 40

# Stepped confidence
python3 train_online_unified.py --env bandit --confidence_type stepped --confidence_start 0.3 --max_position 250

# Constant confidence
python3 train_online_unified.py --env bandit --confidence_type constant --confidence_value 0.8
```

### Evaluation
```bash
# Use the same parameters as training
python3 eval_online.py --env bandit --confidence_type linear --confidence_start 0.3 --epoch 50
```

## File Organization

Results are now organized by confidence function type:
```
figs/
├── evals_online_models/
│   ├── linear/           # Linear confidence results
│   ├── exponential/      # Exponential confidence results
│   ├── stepped/          # Stepped confidence results
│   ├── constant/         # Constant confidence results
│   └── standard/         # Non-confidence results
```

## Batch Files

### Updated Files
- `run_online_bandit_confidence.sbatch` → Uses linear confidence
- `run_online_bandit_confidence_exp.sbatch` → Uses exponential confidence
- `run_online_linear_bandit_confidence.sbatch` → Uses linear confidence
- `run_online_darkroom_confidence.sbatch` → Uses linear confidence
- `run_online_miniworld_confidence.sbatch` → Uses linear confidence
- `eval.sbatch` → Uses linear confidence

### New Files
- `run_online_bandit_confidence_stepped.sbatch` → Uses stepped confidence
- `run_online_bandit_unified.sbatch` → Configurable unified batch file

## Model File Naming

Models are saved with descriptive suffixes:
- Linear: `_linear_start0.3`
- Exponential: `_exponential_lambda40`
- Stepped: `_stepped_start0.3_maxpos250`
- Constant: `_constant_val1.0`

Example: `bandit_shufTrue_lr0.0001_do0_embd32_layer4_head4_envs2000_hists1_samples1000000_var0.3_cov0.0_H500_d5_seed1_linear_start0.3_online_unified.pt`

## Backward Compatibility

The eval script maintains backward compatibility with old model files:
- `--confidence` flag maps to `--confidence_type linear`
- `--confidence_exp` flag maps to `--confidence_type exponential`
