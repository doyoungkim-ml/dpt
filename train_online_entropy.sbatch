#!/bin/bash
#SBATCH --job-name=online_entropy
#SBATCH --output=logs/online_entropy_%j.out
#SBATCH --error=logs/online_entropy_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Get the config file path
CONFIG_FILE="$1"
EVAL_CONFIG_FILE="$2"

# OpenBLAS Warning : Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP=1 option.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run offline training with pre-computed Thompson Sampling distributions
echo "Starting training with config: $CONFIG_FILE"
singularity exec --nv --overlay /scratch/dk5268/pytorch-example/dpt.ext3:ro /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash -c "source /ext3/env.sh; python train_online_entropy_penalty.py --config '$CONFIG_FILE'"


# Get the environment type from config
ENV=$(grep "^env:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')

# Find the model directory (most recent one for this env)
MODEL_DIR=$(ls -td models/$ENV/*/ | head -n 1)

if [ -d "$MODEL_DIR" ]; then
    echo "Training completed. Model saved to: $MODEL_DIR"

    # Run evaluation on the final model
    echo "Starting evaluation..."
    sbatch eval_offline.sbatch $EVAL_CONFIG_FILE $MODEL_DIR/final_model.pt
    sbatch eval_offline_mass.sbatch $EVAL_CONFIG_FILE $MODEL_DIR
else
    echo "Warning: Could not find model directory for evaluation"
fi


echo "All tasks completed"
