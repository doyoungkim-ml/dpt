#!/bin/bash

# Bash script for offline training (no SLURM)
# Usage: ./train_offline.sh <config_file>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 configs/bandit_offline.yaml"
    exit 1
fi

CONFIG_FILE=$1

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting offline training with config: $CONFIG_FILE"

# Run offline training
python train.py --config "$CONFIG_FILE"

echo "Offline training completed"