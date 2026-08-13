#!/bin/bash
# Serve the release checkpoint on one node and look at what it generates.
#
# Phase 2's rollout answers nothing and runs every response to the token limit,
# on weights a post-update comparison certifies as correct. That leaves two
# possibilities this job separates: the MXFP4 serving path is already degenerate
# before any weight update, or it only becomes so afterwards.
#
# No weight update happens here. With SGLANG_PATCH_FILE empty this is the
# control — stock SGLang, the checkpoint as shipped, one engine's worth of GPUs.
# Setting it to the patches a run carries answers the other half: whether those
# patches already change what the initial load serves.
#SBATCH --job-name=miles-dsv4-serve-smoke
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --switches=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

set -euo pipefail

base=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl
code=/home/lbo/work/dsv4_flash_rl/miles
model=${MODEL_NAME:-DeepSeek-V4-Flash-0731}
# Matches the rollout phase 2 configures: packed MXFP4 experts against MXFP8
# activations. Set to 0 to serve the same checkpoint with the experts
# dequantized at load, which is the other half of the bisection.
fp4_experts=${SGLANG_DSV4_FP4_EXPERTS:-1}
moe_backend=${SGLANG_MOE_RUNNER_BACKEND:-flashinfer_mxfp4}

export MODEL="/workspace/data/models/${model}"
export FP4_EXPERTS="${fp4_experts}"
export MOE_BACKEND="${moe_backend}"
export SGLANG_PATCH_FILE="${SGLANG_PATCH_FILE:-}"

echo "job=${SLURM_JOB_ID} model=${model} fp4_experts=${fp4_experts} moe_backend=${moe_backend} patches=${SGLANG_PATCH_FILE:-none}"

srun \
  --container-image=radixark/miles:latest \
  --no-container-mount-home \
  --container-mounts="${code}:/workspace/miles,${base}:/workspace/data" \
  --container-workdir=/workspace/miles \
  bash -lc '
    set -euo pipefail
    export HOME=/tmp/miles-serve-smoke-${SLURM_JOB_ID}
    mkdir -p "${HOME}"
    export PYTHONPATH=/workspace/miles:${PYTHONPATH:-}
    export SGLANG_DSV4_FP4_EXPERTS="${FP4_EXPERTS}"
    export SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1
    export SGLANG_OPT_FP8_WO_A_GEMM=0
    export SGLANG_DG_CACHE_DIR_PER_PROCESS=1
    for patch_file in ${SGLANG_PATCH_FILE}; do
      patch --batch --forward -p1 -d /sgl-workspace/sglang <"${patch_file}"
      for target in $(awk "/^\+\+\+ /{ sub(/^b\//, \"\", \$2); print \$2 }" "${patch_file}"); do
        python3 -m py_compile "/sgl-workspace/sglang/${target}"
        echo "verified patched SGLang source: ${target}"
      done
    done
    python3 /workspace/miles/experiments/dsv4_0731/serve_smoke.py
  '
