#!/bin/bash
# Fetch the official DeepSeek-V4-Flash-0731 checkpoint into the shared workspace.
# Runs on the datamover partition so the login node stays free; `hf download`
# resumes from whatever is already present, so the job is safe to resubmit.
#SBATCH --job-name=dsv4-0731-download
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl/logs/download-0731-%j.log

set -euo pipefail

WORKSPACE=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl
MODEL=deepseek-ai/DeepSeek-V4-Flash-0731

srun \
  --container-image=radixark/miles:latest \
  --container-mounts="${WORKSPACE}":/workspace/data \
  --no-container-mount-home \
  bash -c "
    set -euo pipefail
    export HF_HOME=/workspace/data/hf_cache
    export HF_HUB_ENABLE_HF_TRANSFER=1
    hf download ${MODEL} --local-dir /workspace/data/models/DeepSeek-V4-Flash-0731
    echo '--- downloaded ---'
    du -sh /workspace/data/models/DeepSeek-V4-Flash-0731
    ls /workspace/data/models/DeepSeek-V4-Flash-0731 | wc -l
  "
