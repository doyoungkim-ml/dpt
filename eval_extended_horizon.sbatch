#!/bin/bash
#SBATCH --job-name=extended_eval
#SBATCH --output=logs/extended_eval_%j.out
#SBATCH --error=logs/extended_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Load modules and activate environment
source /home/dk5268/miniconda3/etc/profile.d/conda.sh
conda activate dpt

# Create logs directory if it doesn't exist
mkdir -p logs

# Run extended horizon evaluation with config file and model path
python eval_extended_horizon.py --config "$1" --model_path "$2"

echo "Extended horizon evaluation completed"
