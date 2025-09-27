#!/bin/bash

#################################
#      Slurm Configuration      #
#################################

#SBATCH --nodes=1                                   # Number of nodes to use
#SBATCH --job-name=linear_bandit_dpt                # Name of your job
#SBATCH --output=logs/linear_bandit_%j.out          # Log file for stdout
#SBATCH --error=logs/linear_bandit_%j.err           # Log file for stderr
#SBATCH --constraint="l40s"                         # Request L40S GPU
#SBATCH --gres=gpu:1                                # Request 1 GPU
#SBATCH --mem=32G                                   # Memory per node
#SBATCH --time=24:00:00                             # Max runtime
#SBATCH --comment="preemption=yes;requeue=yes"

set -ex

# Capture directory of this sbatch script so we can reference sibling files reliably
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#--------------------------------------------------------------------#
# Resolve project root robustly using SLURM_SUBMIT_DIR                #
#--------------------------------------------------------------------#

# SLURM_SUBMIT_DIR is the directory where sbatch was invoked
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# For decision-pretrained-transformer: assume submit dir is the project root
# or that the sbatch script is in the project root
if [ -f "$SUBMIT_DIR/collect_data.py" ] && [ -f "$SUBMIT_DIR/train.py" ]; then
    PROJECT_ROOT="$SUBMIT_DIR"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs


###########################
####### Environment #######
###########################

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

###########################
###### Training Script ####
###########################

echo "Starting linear bandit training job"
echo "Project root: $PROJECT_ROOT"
echo "Working directory: $(pwd)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Collect data
echo "Collecting data..."
python3 collect_data.py --env linear_bandit --envs 1000000 --H 200 --dim 10 --var 0.3 --cov 0.0 --lin_d 2 --envs_eval 200

# Train
echo "Training model..."
python3 train.py --env linear_bandit --envs 1000000 --H 200 --dim 10 --lin_d 2 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --seed 1

# Evaluate, choose an appropriate epoch
echo "Evaluating model..."
python3 eval.py --env linear_bandit --envs 1000000 --H 200 --dim 10 --lin_d 2 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --epoch 100 --n_eval 200 --seed 1

echo "Linear bandit training job completed!"
