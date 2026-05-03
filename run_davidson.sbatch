#!/bin/bash
#SBATCH --job-name=davidson_full
#SBATCH --output=outputs/davidson_run-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
# Optional: constrain to specific GPU type (uncomment if available on CoolLab)
# SBATCH --constraint=A100

# Run from submission directory
cd $SLURM_SUBMIT_DIR

# Load modules (update to match CoolLab environment)
module load anaconda
module load cuda/11.8

# Activate conda env (create it beforehand as described in README)
source activate trifuse

# Ensure outputs directory exists
mkdir -p outputs

# Fail fast on error
set -euo pipefail

echo "Starting Davidson full run on $(hostname) at $(date)"

# Run the prepare script then the full training; log stdout+stderr
python prepare_davidson.py && python main.py --mode full --data_path dataset_davidson/ 2>&1 | tee outputs/davidson_run.log

echo "Finished at $(date)"
