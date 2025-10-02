#!/bin/bash
#SBATCH --job-name=online_train
#SBATCH --output=logs/online_train_%j.out
#SBATCH --error=logs/online_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Load modules and activate environment
# Create logs directory if it doesn't exist
mkdir -p logs

# Run online training with config file
python train_online_unified.py --config "$1"

echo "Online training completed"