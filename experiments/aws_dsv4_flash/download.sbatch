#!/bin/bash
#SBATCH --job-name=dsv4-download
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-short
#SBATCH --time=04:00:00

set -euo pipefail

base=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl
hf=${HF_CLI:-/home/lbo/work/2026-08-06/srt-slurm/.venv/bin/hf}

if [[ ! -x "${hf}" ]]; then
  echo "Hugging Face CLI is not executable: ${hf}" >&2
  exit 2
fi

export HF_HOME="${base}/hf_cache"
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "${base}"/{models,datasets,hf_cache,logs}

"${hf}" download sgl-project/DeepSeek-V4-Flash-FP8 \
  --local-dir "${base}/models/DeepSeek-V4-Flash-FP8"
"${hf}" download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir "${base}/datasets/dapo-math-17k"
"${hf}" download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir "${base}/datasets/aime-2024"
