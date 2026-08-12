#!/bin/bash
#SBATCH --job-name=miles-dsv4-image-smoke
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --switches=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

set -euo pipefail

base=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl
code=/home/lbo/work/dsv4_flash_rl/miles

srun \
  --container-image=radixark/miles:latest \
  --no-container-mount-home \
  --container-mounts="${code}:/workspace/miles,${base}:/workspace/data" \
  --container-workdir=/workspace/miles \
  bash -lc '
    set -euo pipefail
    export HOME=/tmp/miles-image-smoke-home
    mkdir -p "${HOME}"
    trap '\''rm -rf -- "${HOME}"'\'' EXIT
    export PYTHONPATH=/workspace/miles:${PYTHONPATH:-}
    echo "host=$(hostname) arch=$(uname -m)"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    python3 - <<"PY"
import importlib.metadata
import os

import ray
import torch

print(f"python_arch={os.uname().machine}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"ray={ray.__version__}")
for package in ("sglang", "transformer-engine", "wandb"):
    print(f"{package}={importlib.metadata.version(package)}")
PY
    python3 - <<"PY"
import megatron
import miles

print(f"megatron_paths={list(megatron.__path__)}")
print(f"miles_module={miles.__file__}")
PY
    find / -maxdepth 4 -type d -name "Megatron-LM" -print 2>/dev/null || true
    python3 scripts/run_deepseek_v4.py train --help \
      | grep -E -- "--train-fp8|--rollout-fp8|--num-nodes" \
      || true
    export WANDB_API_KEY="$(tr -d "\r\n" </workspace/data/secrets/wandb.key)"
    wandb login --verify --relogin "$WANDB_API_KEY" >/dev/null
    echo WANDB_LOGIN_OK
  '
