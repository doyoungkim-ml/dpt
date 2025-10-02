#!/bin/bash

#################################
#      Slurm Configuration      #
#################################

#SBATCH --nodes=1                                   # Number of nodes to use
#SBATCH --job-name=online_bandit_unified_dpt        # Name of your job
#SBATCH --output=logs/online_bandit_unified_%j.out  # Log file for stdout
#SBATCH --error=logs/online_bandit_unified_%j.err   # Log file for stderr
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
###### Configuration ######
###########################

# Configuration variables (modify these as needed)
ENV="bandit" #`"bandit", "linear_bandit", "darkroom_heldout"`
ENVS=2000
HORIZON=500
DIM=5
VAR=0.3
COV=0.0
LR=0.0001
LAYER=4
HEAD=4
SEED=1
NUM_EPOCHS=100
N_EVAL=200

# Confidence function parameters (modify for different experiments)
CONFIDENCE_TYPE="linear"        # Options: linear, exponential, stepped, constant
CONFIDENCE_START=0.3            # For linear and stepped
CONFIDENCE_LAMBDA=40            # For exponential
MAX_POSITION=250               # For stepped
CONFIDENCE_VALUE=1.0           # For constant

###########################
###### Training Script ####
###########################

echo "Starting online bandit training job with unified confidence system"
echo "Project root: $PROJECT_ROOT"
echo "Working directory: $(pwd)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Confidence type: $CONFIDENCE_TYPE"

# Build the training command
TRAIN_CMD="python3 train_online_unified.py --env $ENV --envs $ENVS --H $HORIZON --dim $DIM --var $VAR --cov $COV --lr $LR --layer $LAYER --head $HEAD --shuffle --seed $SEED --samples_per_iter 64 --num_epochs $NUM_EPOCHS --debug --confidence_type $CONFIDENCE_TYPE"

# Add confidence-specific parameters
if [[ "$CONFIDENCE_TYPE" == "linear" || "$CONFIDENCE_TYPE" == "stepped" ]]; then
    TRAIN_CMD="$TRAIN_CMD --confidence_start $CONFIDENCE_START"
fi

if [[ "$CONFIDENCE_TYPE" == "exponential" ]]; then
    TRAIN_CMD="$TRAIN_CMD --confidence_lambda $CONFIDENCE_LAMBDA"
fi

if [[ "$CONFIDENCE_TYPE" == "stepped" ]]; then
    TRAIN_CMD="$TRAIN_CMD --max_position $MAX_POSITION"
fi

if [[ "$CONFIDENCE_TYPE" == "constant" ]]; then
    TRAIN_CMD="$TRAIN_CMD --confidence_value $CONFIDENCE_VALUE"
fi

echo "Training command: $TRAIN_CMD"
eval $TRAIN_CMD

# Build evaluation command
EVAL_BASE_CMD="python3 eval_online.py --env $ENV --envs $ENVS --H $HORIZON --dim $DIM --var $VAR --cov $COV --lr $LR --layer $LAYER --head $HEAD --shuffle --n_eval $N_EVAL --seed $SEED --confidence_type $CONFIDENCE_TYPE"

# Add confidence-specific parameters for evaluation
if [[ "$CONFIDENCE_TYPE" == "linear" || "$CONFIDENCE_TYPE" == "stepped" ]]; then
    EVAL_BASE_CMD="$EVAL_BASE_CMD --confidence_start $CONFIDENCE_START"
fi

if [[ "$CONFIDENCE_TYPE" == "exponential" ]]; then
    EVAL_BASE_CMD="$EVAL_BASE_CMD --confidence_lambda $CONFIDENCE_LAMBDA"
fi

if [[ "$CONFIDENCE_TYPE" == "stepped" ]]; then
    EVAL_BASE_CMD="$EVAL_BASE_CMD --max_position $MAX_POSITION"
fi

if [[ "$CONFIDENCE_TYPE" == "constant" ]]; then
    EVAL_BASE_CMD="$EVAL_BASE_CMD --confidence_value $CONFIDENCE_VALUE"
fi

# Evaluate, choose an appropriate epoch
echo "Evaluating model..."
for epoch in {10..100..10}; do
    EVAL_CMD="$EVAL_BASE_CMD --epoch $epoch"
    echo "Evaluation command: $EVAL_CMD"
    eval $EVAL_CMD
done

echo "Online bandit training job with $CONFIDENCE_TYPE confidence completed!"
