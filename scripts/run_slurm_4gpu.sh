#!/bin/bash
#SBATCH --job-name=pipelinerl-4gpu
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=44
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Enable error handling and debugging
set -x -e

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Working directory: $(pwd)"
echo "Starting time: $(date)"

# Load CUDA module
module load cuda/12.4

# Activate environment
source .pipeline-rl/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Get config name from first argument (required)
if [ -z "$1" ]; then
    echo "Error: Config name is required as first argument"
    echo "Usage: sbatch scripts/run_slurm_4gpu.sh <config_name> [additional_args...]"
    echo "Example: sbatch scripts/run_slurm_4gpu.sh guessing_4gpu"
    exit 1
fi

CONFIG_NAME=$1
shift  # Remove first argument

# Launch the training
python -m pipelinerl.launch --config-name ${CONFIG_NAME} output_dir=results/${CONFIG_NAME} "$@"

echo "Ending time: $(date)"
