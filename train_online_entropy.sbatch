#!/bin/bash
#SBATCH --job-name=online_entropy
#SBATCH --output=logs/online_entropy_%j.out
#SBATCH --error=logs/online_entropy_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Load modules and activate environment
# Create logs directory if it doesn't exist
mkdir -p logs

# Run online training with entropy penalty using config file
python train_online_entropy_penalty.py --config "$1"

echo "Online training with entropy penalty completed"

