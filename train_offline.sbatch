#!/bin/bash
#SBATCH --job-name=offline_train
#SBATCH --output=logs/offline_train_%j.out
#SBATCH --error=logs/offline_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Load modules and activate environment

# Create logs directory if it doesn't exist
mkdir -p logs

# create data
python collect_data.py --config "$1"
# Run offline training with config file
python train.py --config "$1"

echo "Offline training completed"