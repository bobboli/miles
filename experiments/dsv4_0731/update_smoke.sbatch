#!/bin/bash
# Serve the release checkpoint on one node, update it with itself, and compare.
#
# Reproduce phase 2's failure on one node by updating the model with its own
# weights. An identity update should be invisible; if generation degrades across
# it, the whole failure is reachable without a trainer, a quantizer, or eight
# nodes.
#SBATCH --job-name=miles-dsv4-update-smoke
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
export UPDATE_SELECTOR="${UPDATE_SELECTOR:-.ffn.experts.}"
export MEMORY_CYCLE="${MEMORY_CYCLE:-1}"
export TRANSPORT="${TRANSPORT:-bucket}"
export BUCKET_BYTES="${BUCKET_BYTES:-67108864}"
export MEM_FRACTION="${MEM_FRACTION:-0.45}"

echo "job=${SLURM_JOB_ID} model=${model} fp4_experts=${fp4_experts} moe_backend=${moe_backend} patches=${SGLANG_PATCH_FILE:-none} selector=${UPDATE_SELECTOR} memory_cycle=${MEMORY_CYCLE} transport=${TRANSPORT}"

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
    python3 /workspace/miles/experiments/dsv4_0731/update_smoke.py
  '
