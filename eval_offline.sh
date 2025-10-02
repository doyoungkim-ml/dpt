#!/bin/bash

# Bash script for offline evaluation (no SLURM)
# Usage: ./eval_offline.sh <config_file> <model_path>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_file> <model_path>"
    echo "Example: $0 configs/bandit_eval.yaml models/bandit/123/epoch100.pt"
    exit 1
fi

CONFIG_FILE=$1
MODEL_PATH=$2

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file '$MODEL_PATH' not found"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting offline evaluation with config: $CONFIG_FILE"
echo "Using model: $MODEL_PATH"

# Run offline evaluation
python eval.py --config "$CONFIG_FILE" --model_path "$MODEL_PATH"

echo "Offline evaluation completed"