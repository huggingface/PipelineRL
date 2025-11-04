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

```bash
# Allocate 8 GPUs on a single node
salloc --partition=hopper-prod --nodes=1 --gpus-per-node=8 --cpus-per-gpu=11 --mem-per-gpu=248G --time=4:00:00

# Once allocated, activate the environment
source .pipeline-rl/bin/activate

# Run PipelineRL
python -m pipelinerl.launch output_dir=results/base1
```

For 4 GPUs:
```bash
salloc --partition=hopper-prod --nodes=1 --gpus-per-node=4 --cpus-per-gpu=11 --mem-per-gpu=248G --time=4:00:00
source .pipeline-rl/bin/activate
python -m pipelinerl.launch --config-name base_4gpu output_dir=results/base1
```

To use Redis instead of the filesystem for data streaming:
```bash
python -m pipelinerl.launch streams=redis output_dir=results/base1
```

When done, exit the allocation:
```bash
exit
```

### SLURM cluster mode

This cluster uses the `hopper-prod` partition with H100 GPUs (8 GPUs per node).

#### Single node with 8 GPUs (default)

```bash
# Create logs directory
mkdir -p logs

# Submit the job
sbatch scripts/run_slurm.sh
```

You can pass additional arguments to the launch script:
```bash
sbatch scripts/run_slurm.sh streams=redis model_path=Qwen/Qwen2.5-32B
```

#### Single node with 4 GPUs

```bash
sbatch scripts/run_slurm_4gpu.sh
```

#### Multi-node training

For multi-node training (e.g., 2 nodes with 8 GPUs each = 16 GPUs total):
```bash
sbatch --nodes=2 scripts/run_slurm.sh
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
