#!/bin/bash

#SBATCH --cpus-per-task=1      # Use 1 CPU cores for data loading
#SBATCH --mem=64G               # 32 GB of RAM
#SBATCH --time=72:00:00         # Up to 72 hours of training

# The cluster will send an email to this address if the job fails or ends
#SBATCH --mail-user=s.sourbron@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

# Assigns an internal “comment” (or name) to the job in the scheduler
#SBATCH --comment=corr

# Assign a name to the job
#SBATCH --job-name=corr

# Write logs to the logs folder
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Unsets the CPU binding policy.
# Some clusters automatically bind threads to cores; unsetting it can 
# prevent performance issues if your code manages threading itself 
# (e.g. OpenMP, NumPy, or PyTorch).
unset SLURM_CPU_BIND

# Ensures that all your environment variables from the submission 
# environment are passed into the job’s environment
export SLURM_EXPORT_ENV=ALL

# Loads the Anaconda module provided by the cluster.
# (On HPC systems, software is usually installed as “modules” to avoid version conflicts.)
module load Anaconda3/2024.02-1
module load Python/3.10.8-GCCcore-12.2.0 # essential to load latest GCC
#module load CUDA/11.8.0 # must match with version in environment.yml

# Initialize Conda for this non-interactive shell
eval "$(conda shell.bash hook)"

# Activates your Conda environment named venv.
# (Older clusters use source activate; newer Conda versions use conda activate venv.)
# We assume that the conda environment 'venv' has already been created
conda activate corr

# Get the current username
USERNAME=$(whoami)

# Define path variables here
BASE_DIR="/mnt/parscratch/users/$USERNAME/iBEAt_Build/kidneyvol"
CODE="$BASE_DIR/iBEAt-pipeline-kidneyvol"
DATA="$BASE_DIR/stage_7_normalized"
BUILD="$BASE_DIR/stage_7_shape_analysis"

# srun runs your program on the allocated compute resources managed by Slurm
srun /users/md1spsx/.conda/envs/corr/bin/python "$CODE/src/stage_7_corr.py" --data="$DATA" --build="$BUILD"

