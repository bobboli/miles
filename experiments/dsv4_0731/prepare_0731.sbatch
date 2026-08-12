#!/bin/bash
# Cast DeepSeek-V4-Flash-0731 to BF16, then quantize that cast to MXFP8 for rollout.
# The release ships MXFP4 routed experts alongside block-scaled FP8 elsewhere, so
# the BF16 cast is what both the Megatron trainer and the MXFP8 rollout derive from.
# Both stages skip themselves when their output index already exists.
#SBATCH --job-name=dsv4-0731-prepare
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
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"model_loading","description":"DeepSeek-V4-Flash-0731 MXFP4/FP8 to BF16 cast and MXFP8 quantization."}}'

set -euo pipefail

base=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl
code=/home/lbo/work/dsv4_flash_rl/miles
model_name=${MODEL_NAME:-DeepSeek-V4-Flash-0731}

export MODEL_NAME="${model_name}"
export SLURM_EXPORT_ENV=ALL

echo "job=${SLURM_JOB_ID} model=${model_name}"

srun --kill-on-bad-exit=1 \
  --container-image=radixark/miles:latest \
  --no-container-mount-home \
  --container-mounts="${code}:/workspace/miles,${base}:/workspace/data" \
  --container-workdir=/workspace/miles \
  bash -lc '
    set -euo pipefail
    export HOME=/tmp/miles-dsv4-prepare-home-${SLURM_JOB_ID}
    export PYTHONPATH=/workspace/miles:${PYTHONPATH:-}
    export HF_HOME=/workspace/data/hf_cache
    mkdir -p "${HOME}" /workspace/data/{models,datasets,hf_cache}

    python3 scripts/run_deepseek_v4.py prepare-single \
      --model-name "${MODEL_NAME}" \
      --hardware GB300 \
      --hf-checkpoint /workspace/data/models/"${MODEL_NAME}" \
      --model-dir /workspace/data/models \
      --data-dir /workspace/data/datasets

    python3 experiments/dsv4_0731/assert_fully_dequantized.py \
      --model-dir /workspace/data/models/"${MODEL_NAME}"-bf16

    python3 scripts/run_deepseek_v4.py prepare-mxfp8 \
      --model-name "${MODEL_NAME}" \
      --hardware GB300 \
      --no-rollout-fp8 --rollout-mxfp8 \
      --model-dir /workspace/data/models \
      --data-dir /workspace/data/datasets

    du -sh /workspace/data/models/"${MODEL_NAME}"{,-bf16,-MXFP8}
  '
