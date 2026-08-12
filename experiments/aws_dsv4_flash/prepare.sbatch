#!/bin/bash
#SBATCH --job-name=dsv4-prepare
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --switches=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --comment={"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"model_loading","description":"DeepSeek-V4-Flash checkpoint download and single-node FP8-to-BF16 cast."}}

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
    export HOME=/tmp/miles-dsv4-prepare-home
    export PYTHONPATH=/workspace/miles:${PYTHONPATH:-}
    export HF_HOME=/workspace/data/hf_cache
    mkdir -p "${HOME}" /workspace/data/{models,datasets,hf_cache}

    python3 scripts/run_deepseek_v4.py prepare-download \
      --model-name DeepSeek-V4-Flash-FP8 \
      --hardware GB300 \
      --model-dir /workspace/data/models \
      --data-dir /workspace/data/datasets

    python3 scripts/run_deepseek_v4.py prepare-single \
      --model-name DeepSeek-V4-Flash-FP8 \
      --hardware GB300 \
      --hf-checkpoint /workspace/data/models/DeepSeek-V4-Flash-FP8 \
      --model-dir /workspace/data/models \
      --data-dir /workspace/data/datasets
  '
