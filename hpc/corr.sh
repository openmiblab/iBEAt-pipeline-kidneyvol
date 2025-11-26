#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1               # one task (one job process)
#SBATCH --cpus-per-task=8       # 64 CPU cores for that task
#SBATCH --mem=64G                # 64 GB for the whole node (default is per node)
#SBATCH --time=72:00:00

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
PROJ="/mnt/parscratch/users/$USERNAME/iBEAt_Build/kidneyvol"

# srun runs your program on the allocated compute resources managed by Slurm
srun /users/md1spsx/.conda/envs/corr/bin/python "$PROJ/iBEAt-pipeline-kidneyvol/src/stage_8_build_features.py" --data="$PROJ"
srun /users/md1spsx/.conda/envs/corr/bin/python "$PROJ/iBEAt-pipeline-kidneyvol/src/stage_9_build_correlation_matrices.py" --data="$PROJ"