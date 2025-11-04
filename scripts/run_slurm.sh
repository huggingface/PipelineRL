#!/bin/bash
#SBATCH --job-name=pipelinerl-8gpu
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=88
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
    echo "Usage: sbatch scripts/run_slurm.sh <config_name> [additional_args...]"
    echo "Example: sbatch scripts/run_slurm.sh math"
    exit 1
fi

CONFIG_NAME=$1
shift  # Remove first argument

# Set environment variables for multi-node
export WORLD_SIZE=$SLURM_JOB_NUM_NODES
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch the training
# For single node (default)
if [ $SLURM_JOB_NUM_NODES -eq 1 ]; then
    python -m pipelinerl.launch --config-name ${CONFIG_NAME} output_dir=results/${CONFIG_NAME} "$@"
else
    # For multi-node setup
    srun --nodes=$SLURM_JOB_NUM_NODES \
         --ntasks-per-node=1 \
         bash -c "
            export RANK=\$SLURM_PROCID
            export LOCAL_RANK=\$SLURM_LOCALID
            export WORLD_SIZE=$WORLD_SIZE
            export MASTER_ADDR=$MASTER_ADDR
            export MASTER_PORT=$MASTER_PORT
            python -m pipelinerl.launch --config-name ${CONFIG_NAME} output_dir=results/${CONFIG_NAME} $@
         "
fi

echo "Ending time: $(date)"
