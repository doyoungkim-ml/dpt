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

# Create logs directory if it doesn't exist
mkdir -p logs

# Run offline evaluation with config file and model path
singularity exec --nv --overlay /scratch/dk5268/pytorch-example/dpt.ext3:ro /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash -c "source /ext3/env.sh; python eval.py --config '$1' --model_path '$2'"

echo "Offline evaluation completed"