#!/bin/bash
#SBATCH --job-name=offline_mass_eval
#SBATCH --output=logs/offline_mass_eval_%j.out
#SBATCH --error=logs/offline_mass_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Usage: sbatch eval_offline_mass.sbatch <config_file> <checkpoint_directory>
# Example: sbatch eval_offline_mass.sbatch configs/bandit.yaml models/bandit/135081

# Load modules and activate environment

# Create logs directory if it doesn't exist
mkdir -p logs

# Run mass evaluation with config file and checkpoint directory
python eval_mass.py --config "$1" --checkpoint_dir "$2"

echo "Mass evaluation completed"
mass_eval_dir=figs/mass_eval_$2
cp $2/config.yaml $mass_eval_dir/config.yaml
cp $2/logs.txt $mass_eval_dir/logs.txt
cp $2/train_loss.png $mass_eval_dir/train_loss.png
