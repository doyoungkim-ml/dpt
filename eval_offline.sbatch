#!/bin/bash
#SBATCH --job-name=offline_eval
#SBATCH --output=logs/offline_eval_%j.out
#SBATCH --error=logs/offline_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Load modules and activate environment
module load python/3.9
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run offline evaluation with config file and model path
python eval.py --config $1 --model_path $2

echo "Offline evaluation completed"