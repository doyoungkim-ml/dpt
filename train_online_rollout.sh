#!/bin/bash

# Bash script for online rollout training (no SLURM)
# Usage: ./train_online_rollout.sh <config_file>
#
# This script trains a model by collecting data at each epoch using the current model
# (or random initialization for the first epoch), then training on that collected data.

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

echo "Starting online rollout training with config: $CONFIG_FILE"
echo "Note: Data will be collected at each epoch using the current model"

# Run online rollout training (data collection happens inside the training script)
python train_online_rollout.py --config "$CONFIG_FILE"

echo "Online rollout training completed"

