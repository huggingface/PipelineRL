#!/bin/bash
#SBATCH --job-name=pipelinerl
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=11
#SBATCH --mem-per-gpu=248G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

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

# Set environment variables for multi-node
export WORLD_SIZE=$SLURM_JOB_NUM_NODES
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch the training
# For single node (default)
if [ $SLURM_JOB_NUM_NODES -eq 1 ]; then
    python -m pipelinerl.launch output_dir=results/base1 "$@"
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
            python -m pipelinerl.launch output_dir=results/base1 $@
         "
fi

echo "Ending time: $(date)"
