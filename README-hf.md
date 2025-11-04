# Pipeline RL: Setup with uv

This guide provides instructions for setting up PipelineRL using `uv` instead of conda.

## Setup

Clone the repository and change the directory to `pipelinerl`:
```bash
git clone https://github.com/huggingface/PipelineRL.git
cd PipelineRL
```

Create a virtual environment with Python 3.11:
```bash
uv venv --python 3.11 .pipeline-rl
```

Activate the environment:
```bash
source .pipeline-rl/bin/activate  # Linux/macOS
# or on Windows: .pipeline-rl\Scripts\activate
```

Install dependencies:
```bash
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install setuptools
uv pip install -e . --no-build-isolation
```

### Redis streaming backend (optional)

By default Pipeline-RL will use the file system as the medium for streaming the generated data to the trainer processes. This works on one node, but the files can get quite large. To use Redis instead, you need to install the Redis server.

Install the redis-server Python package:
```bash
uv pip install redis-server
```

Note: The `redis-server` PyPI package only supports Redis versions 5.0.7 and 6.0rc2, and officially supports Python 3.5-3.9. Since this project uses Python 3.11, compatibility may vary.

The Python redis client library should already be included in the project dependencies, so you only need to ensure the Redis server is running.

## Run experiments

### Interactive mode (on SLURM cluster)

For interactive development and debugging, first allocate compute resources:

#### Option 1: Using salloc (get an interactive shell on a compute node)
```bash
# Allocate 8 GPUs on a single node
salloc --partition=hopper-prod --nodes=1 --gpus-per-node=8 --cpus-per-gpu=11 --mem-per-gpu=248G --time=4:00:00

# Once allocated, load CUDA and activate the environment
module load cuda/12.4
source .pipeline-rl/bin/activate

# Run PipelineRL (use guessing config for testing)
python -m pipelinerl.launch --config-name guessing output_dir=results/guessing
```

#### Option 2: Using srun (run command directly on compute node)
```bash
# Run a single command on a GPU node
srun --partition=hopper-prod --nodes=1 --gpus-per-node=8 --cpus-per-gpu=11 --mem-per-gpu=248G --time=4:00:00 \
  bash -c "module load cuda/12.4 && source .pipeline-rl/bin/activate && python -m pipelinerl.launch --config-name guessing output_dir=results/guessing"
```

#### For 4 GPUs (using salloc):
```bash
salloc --partition=hopper-prod --nodes=1 --gpus-per-node=4 --cpus-per-gpu=11 --mem-per-gpu=248G --time=4:00:00
module load cuda/12.4
source .pipeline-rl/bin/activate
python -m pipelinerl.launch --config-name guessing output_dir=results/guessing
```

#### For 4 GPUs (using srun):
```bash
srun --partition=hopper-prod --nodes=1 --gpus-per-node=4 --cpus-per-gpu=11 --mem-per-gpu=248G --time=4:00:00 \
  bash -c "module load cuda/12.4 && source .pipeline-rl/bin/activate && python -m pipelinerl.launch --config-name guessing output_dir=results/guessing"
```

To use Redis instead of the filesystem for data streaming:
```bash
python -m pipelinerl.launch --config-name guessing streams=redis output_dir=results/guessing
```

When done, exit the allocation:
```bash
exit
```

### SLURM cluster mode

This cluster uses the `hopper-prod` partition with H100 GPUs (8 GPUs per node).

#### Single node with 8 GPUs

```bash
# Create logs directory
mkdir -p logs

# Submit the job (config name is required)
sbatch scripts/run_slurm.sh math

# Use different configs
sbatch scripts/run_slurm.sh guessing

# Pass additional Hydra overrides
sbatch scripts/run_slurm.sh math streams=redis
```

**Note:** The config name is **required** as the first argument. The script automatically uses `results/<config_name>` as the output directory.

### Understanding GPU Allocation

PipelineRL automatically divides GPUs based on `actor_fraction` and `finetune_fraction` settings:

**To override GPU allocation:**
```bash
# Custom allocation: 1 GPU for actors, 3 GPUs for finetuning
python -m pipelinerl.launch --config-name guessing_4gpu world.actor_fraction=2 world.finetune_fraction=6 output_dir=results/custom
```

#### Single node with 4 GPUs

```bash
# Submit with config name (required)
sbatch scripts/run_slurm_4gpu.sh math

# Use different configs
sbatch scripts/run_slurm_4gpu.sh guessing

# Pass additional Hydra overrides
sbatch scripts/run_slurm_4gpu.sh guessing world.actor_fraction=1 world.finetune_fraction=3
```

#### Multi-node training

For multi-node training (e.g., 2 nodes with 8 GPUs each = 16 GPUs total):
```bash
# Config name is required
sbatch --nodes=2 scripts/run_slurm.sh math

# Different config
sbatch --nodes=2 scripts/run_slurm.sh guessing

# With additional overrides
sbatch --nodes=2 scripts/run_slurm.sh math streams=redis
```

The script automatically detects multi-node setup and configures the environment variables (`WORLD_SIZE`, `RANK`, `MASTER_ADDR`) required by PipelineRL.

#### Monitor your job

Check job status:
```bash
squeue -u $USER
```

View live output:
```bash
tail -f logs/slurm-<job_id>.out
```

Cancel a job:
```bash
scancel <job_id>
```

#### Customize SLURM parameters

You can modify the SLURM scripts in `scripts/` directory to customize:
- `--time`: Job time limit (default: 48:00:00)
- `--partition`: Partition name (default: hopper-prod)
- `--gpus-per-node`: Number of GPUs per node
- `--mem-per-gpu`: Memory per GPU (default: 248G)

Or override them when submitting:
```bash
sbatch --time=72:00:00 --partition=hopper-extra scripts/run_slurm.sh
```
