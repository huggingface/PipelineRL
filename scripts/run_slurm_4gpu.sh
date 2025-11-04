#!/bin/bash
#SBATCH --job-name=pipelinerl-4gpu
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
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

# Activate environment
source .pipeline-rl/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch the training
python -m pipelinerl.launch --config-name base_4gpu output_dir=results/base1 "$@"

echo "Ending time: $(date)"
