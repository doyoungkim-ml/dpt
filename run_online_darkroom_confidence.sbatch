#!/bin/bash

#################################
#      Slurm Configuration      #
#################################

#SBATCH --nodes=1                                   # Number of nodes to use
#SBATCH --job-name=online_darkroom_confidence_dpt   # Name of your job
#SBATCH --output=logs/online_darkroom_confidence_%j.out        # Log file for stdout
#SBATCH --error=logs/online_darkroom_confidence_%j.err         # Log file for stderr
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

echo "Starting online darkroom training job with confidence"
echo "Project root: $PROJECT_ROOT"
echo "Working directory: $(pwd)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Train online with confidence (no data collection needed)
echo "Training online model with confidence..."
python3 train_online_confidence.py --env darkroom_heldout --envs 10000 --H 100 --dim 10 --lr 0.001 --layer 4 --head 4 --shuffle --seed 1 --samples_per_iter 64 --num_epochs 10 --confidence_start 0.3

# Evaluate, choose an appropriate epoch
echo "Evaluating model..."
for epoch in {1..10}; do
    python3 eval_online.py --env darkroom_heldout --envs 10000 --H 100 --dim 10 --lr 0.001 --layer 4 --head 4 --shuffle --epoch $epoch --seed 1
done

echo "Online darkroom training job with confidence completed!"
