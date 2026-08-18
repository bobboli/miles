#!/bin/bash
#SBATCH --job-name=miles-dsv4-rl
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl/logs/%x-%j.log
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=8
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --switches=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"model_loading","description":"DeepSeek-V4-Flash model loading, CUDA graph compilation, and RL validation."}}'

set -euo pipefail

base=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl
code=${CODE:-/home/lbo/work/dsv4_flash_rl/miles}
megatron_bridge_path=${MEGATRON_BRIDGE_PATH:-}
precision=${PRECISION:-fp8}
model_name=${MODEL_NAME:-DeepSeek-V4-Flash-FP8}
init_model_source=${INIT_MODEL_SOURCE:-torch_dist}
rollout_weight_source=${ROLLOUT_WEIGHT_SOURCE:-checkpoint}
num_rollout=${NUM_ROLLOUT:-2}
update_weight_buffer_size=${UPDATE_WEIGHT_BUFFER_SIZE:-1073741824}
sglang_mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC:-0.7}
nccl_algo=${NCCL_ALGO:-^NVLS}
nccl_launch_order_implicit=${NCCL_LAUNCH_ORDER_IMPLICIT:-1}
optimizer_offload=${OPTIMIZER_OFFLOAD:-0}
rematerialize_param=${REMATERIALIZE_PARAM_FROM_MASTER_WEIGHT:-1}
offload_train_target=${OFFLOAD_TRAIN_TARGET:-cpu}
offload_train_disk_dir=${OFFLOAD_TRAIN_DISK_DIR:-/raid/scratch/lbo-miles-train-offload-${SLURM_JOB_ID}}
offload_train_disk_chunk_mb=${OFFLOAD_TRAIN_DISK_CHUNK_MB:-256}
mode=${MODE:-debug_minimal}
enable_eval=${ENABLE_EVAL:-0}
enable_r3=${ENABLE_R3:-1}
log_passrate=${LOG_PASSRATE:-0}
wandb_run_id=${WANDB_RUN_ID:-}
skip_saving=${SKIP_SAVING:-1}
use_fault_tolerance=${USE_FAULT_TOLERANCE:-1}
sglang_patch_file=${SGLANG_PATCH_FILE:-}
# The SGLang patches are applied to the image's own sources, so a moving tag can
# break them without any change here. Pin the digest for a result worth keeping.
container_image=${CONTAINER_IMAGE:-radixark/miles:latest}
moe_runner_backend=${SGLANG_MOE_RUNNER_BACKEND:-}
fp8_gemm_backend=${SGLANG_FP8_GEMM_BACKEND:-}
# Appended verbatim to --extra-args, for switches this wrapper does not model.
extra_args_append=${MILES_EXTRA_ARGS_APPEND:-}
run_id=${RUN_ID:-dsv4-aws-${precision}-$(date -u +%Y%m%dT%H%M%SZ)}

optimizer_offload_args="--no-optimizer-offload"
if [[ "${optimizer_offload}" == 1 ]]; then
  optimizer_offload_args="--optimizer-offload"
fi

rematerialize_args="--rematerialize-param-from-master-weight"
if [[ "${rematerialize_param}" == 0 ]]; then
  rematerialize_args=""
fi

case "${precision}" in
  fp8)
    precision_args="--train-fp8 --rollout-fp8"
    train_precision_extra_args=""
    ;;
  bf16)
    precision_args="--no-train-fp8 --rollout-fp8"
    train_precision_extra_args="--bf16 --transformer-impl transformer_engine"
    ;;
  mxfp8)
    # Both sides read the MXFP8 checkpoint converted from the BF16 cast.
    precision_args="--no-train-fp8 --train-mxfp8 --no-rollout-fp8 --rollout-mxfp8"
    train_precision_extra_args=""
    ;;
  mxfp8-fp4r)
    # Rollout serves the release checkpoint as shipped: MXFP4 routed experts
    # against MXFP8 activations, while training stays on the MXFP8 recipe.
    precision_args="--no-train-fp8 --train-mxfp8 --rollout-fp8"
    train_precision_extra_args=""
    ;;
  *)
    echo "PRECISION must be fp8, bf16, mxfp8 or mxfp8-fp4r, got: ${precision}" >&2
    exit 2
    ;;
esac

eval_args="--enable-eval"
if [[ "${enable_eval}" == 0 ]]; then
  eval_args="--no-enable-eval"
fi

r3_args="--enable-r3"
if [[ "${enable_r3}" == 0 ]]; then
  r3_args="--no-enable-r3"
fi

log_passrate_args=""
if [[ "${log_passrate}" == 1 ]]; then
  log_passrate_args="--log-passrate"
fi

wandb_run_id_args=""
if [[ -n "${wandb_run_id}" ]]; then
  wandb_run_id_args="--wandb-run-id ${wandb_run_id}"
fi

save_args=""
if [[ "${skip_saving}" == 1 ]]; then
  save_args="--skip-saving"
fi

# Empty keeps whatever backend the precision choice implies.
backend_args=""
if [[ -n "${moe_runner_backend}" ]]; then
  backend_args+=" --sglang-moe-runner-backend ${moe_runner_backend}"
fi
if [[ -n "${fp8_gemm_backend}" ]]; then
  backend_args+=" --sglang-fp8-gemm-backend ${fp8_gemm_backend}"
fi

fault_tolerance_args="--use-fault-tolerance"
if [[ "${use_fault_tolerance}" == 0 ]]; then
  fault_tolerance_args="--no-use-fault-tolerance"
fi

mkdir -p "${base}/runs/${run_id}/ray" "${base}/logs" "${base}/hf_cache"
rm -f "${base}/runs/${run_id}/ray/"{ready-*,done}

container_mounts="${code}:/workspace/miles,${base}:/workspace/data,/raid:/raid"
if [[ -n "${megatron_bridge_path}" ]]; then
  [[ -d "${megatron_bridge_path}/src/megatron/bridge" ]]
  container_mounts+=",${megatron_bridge_path}:/workspace/Megatron-Bridge"
fi

export BASE="${base}"
export CODE="${code}"
export PRECISION="${precision}"
export MODEL_NAME="${model_name}"
export INIT_MODEL_SOURCE="${init_model_source}"
export ROLLOUT_WEIGHT_SOURCE="${rollout_weight_source}"
export PRECISION_ARGS="${precision_args}"
export TRAIN_PRECISION_EXTRA_ARGS="${train_precision_extra_args}"
export NUM_ROLLOUT="${num_rollout}"
export UPDATE_WEIGHT_BUFFER_SIZE="${update_weight_buffer_size}"
export SGLANG_MEM_FRACTION_STATIC="${sglang_mem_fraction_static}"
export NCCL_ALGO="${nccl_algo}"
export NCCL_LAUNCH_ORDER_IMPLICIT="${nccl_launch_order_implicit}"
export OPTIMIZER_OFFLOAD_ARGS="${optimizer_offload_args}"
export REMATERIALIZE_ARGS="${rematerialize_args}"
export OFFLOAD_TRAIN_TARGET="${offload_train_target}"
export OFFLOAD_TRAIN_DISK_DIR="${offload_train_disk_dir}"
export OFFLOAD_TRAIN_DISK_CHUNK_MB="${offload_train_disk_chunk_mb}"
export MODE="${mode}"
export EVAL_ARGS="${eval_args}"
export R3_ARGS="${r3_args}"
export LOG_PASSRATE_ARGS="${log_passrate_args}"
export WANDB_RUN_ID_ARGS="${wandb_run_id_args}"
export SAVE_ARGS="${save_args}"
export FAULT_TOLERANCE_ARGS="${fault_tolerance_args}"
export BACKEND_ARGS="${backend_args}"
export EXTRA_ARGS_APPEND="${extra_args_append}"
export SGLANG_PATCH_FILE="${sglang_patch_file}"
export CONTAINER_IMAGE="${container_image}"
export RUN_ID="${run_id}"
export HEAD_NODE
HEAD_NODE=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)
# Nested srun steps need variables computed by this wrapper even when sbatch
# was submitted with an explicit --export list.
export SLURM_EXPORT_ENV=ALL

echo "job=${SLURM_JOB_ID} run_id=${run_id} model=${model_name} precision=${precision} init_model_source=${init_model_source} rollout_weight_source=${rollout_weight_source} nodes=${SLURM_JOB_NUM_NODES}"
echo "head_node=${HEAD_NODE} code=${code} megatron_bridge_path=${megatron_bridge_path:-image-default} num_rollout=${num_rollout} update_weight_buffer_size=${update_weight_buffer_size} sglang_mem_fraction_static=${sglang_mem_fraction_static} nccl_algo=${nccl_algo} nccl_launch_order_implicit=${nccl_launch_order_implicit} optimizer_offload=${optimizer_offload} rematerialize_param=${rematerialize_param} offload_train_target=${offload_train_target} offload_train_disk_dir=${offload_train_disk_dir} mode=${mode} eval=${enable_eval} r3=${enable_r3} log_passrate=${log_passrate} skip_saving=${skip_saving} use_fault_tolerance=${use_fault_tolerance} wandb_run_id=${wandb_run_id:-auto} sglang_patch_file=${sglang_patch_file:-none} container_image=${container_image}"

srun --kill-on-bad-exit=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --no-container-mount-home \
  --container-mounts="${container_mounts}" \
  --container-workdir=/workspace/miles \
  bash -lc '
    set -euo pipefail

    rank=${SLURM_PROCID}
    ray_dir=/workspace/data/runs/${RUN_ID}/ray
    export HOME=/tmp/miles-home-${SLURM_JOB_ID}-${rank}
    mkdir -p "${HOME}"
    chmod 700 "${HOME}"
    export PYTHONPATH=/workspace/miles:/workspace/Megatron-Bridge/src:${PYTHONPATH:-}
    export LD_PRELOAD=/workspace/data/lib/libidempotent_shm_unlink.so${LD_PRELOAD:+:${LD_PRELOAD}}
    if [[ -n "${SGLANG_PATCH_FILE}" ]]; then
      # Space-separated so an experiment can stack patches without merging them.
      for patch_file in ${SGLANG_PATCH_FILE}; do
        patch --batch --forward -p1 -d /sgl-workspace/sglang <"${patch_file}"
        # A patch that no longer applies fails the job rather than silently
        # reverting, and every file it touched has to stay importable.
        for target in $(awk "/^\+\+\+ /{ sub(/^b\//, \"\", \$2); print \$2 }" "${patch_file}"); do
          python3 -m py_compile "/sgl-workspace/sglang/${target}"
          echo "verified patched SGLang source: ${target}"
        done
      done
    fi
    [[ -n "${MILES_NCCL_DEBUG:-}" ]] && export NCCL_DEBUG="${MILES_NCCL_DEBUG}"
    [[ -n "${MILES_NCCL_DEBUG_SUBSYS:-}" ]] && export NCCL_DEBUG_SUBSYS="${MILES_NCCL_DEBUG_SUBSYS}"
    [[ -n "${MILES_NCCL_LAUNCH_RACE_FATAL:-}" ]] && export NCCL_LAUNCH_RACE_FATAL="${MILES_NCCL_LAUNCH_RACE_FATAL}"
    head_ip=$(getent ahostsv4 "${HEAD_NODE}" | awk "NR == 1 { print \$1 }")
    node_ip=$(getent ahostsv4 "$(hostname)" | awk "NR == 1 { print \$1 }")

    wandb_key=$(tr -d "\r\n" </workspace/data/secrets/wandb.key)
    wandb login --verify --relogin "${wandb_key}" >/dev/null
    unset wandb_key WANDB_API_KEY

    finish() {
      if [[ "${rank}" == 0 ]]; then
        touch "${ray_dir}/done"
      fi
      ray stop --force >/dev/null 2>&1 || true
      rm -rf -- "${HOME}"
    }
    trap finish EXIT

    ray stop --force >/dev/null 2>&1 || true
    # Ray can leave SGLang multiprocessing children behind after an interrupted
    # job. Remove their fixed-port listeners before starting a new cluster on
    # an exclusive node.
    pkill -9 -f "[s]glang::" >/dev/null 2>&1 || true
    for port in 15000 15001 15002 15003; do
      stale_pids=$(lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)
      if [[ -n "${stale_pids}" ]]; then
        kill -9 ${stale_pids} >/dev/null 2>&1 || true
      fi
    done
    if [[ "${rank}" == 0 ]]; then
      ray start --head \
        --node-ip-address="${node_ip}" \
        --num-gpus=4 \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265
    else
      for _ in $(seq 1 120); do
        timeout 1 bash -c "</dev/tcp/${head_ip}/6379" 2>/dev/null && break
        sleep 2
      done
      ray start \
        --address="${head_ip}:6379" \
        --node-ip-address="${node_ip}" \
        --num-gpus=4 \
        --disable-usage-stats
    fi

    touch "${ray_dir}/ready-${rank}"

    if [[ "${rank}" == 0 ]]; then
      for _ in $(seq 1 180); do
        ready=$(find "${ray_dir}" -maxdepth 1 -name "ready-*" | wc -l)
        [[ "${ready}" == "${SLURM_JOB_NUM_NODES}" ]] && break
        sleep 2
      done
      ready=$(find "${ray_dir}" -maxdepth 1 -name "ready-*" | wc -l)
      [[ "${ready}" == "${SLURM_JOB_NUM_NODES}" ]]
      ray status

      export MASTER_ADDR="${head_ip}"
      export MILES_SCRIPT_EXTERNAL_RAY=1
      export HF_HOME=/workspace/data/hf_cache

      # Precision, evaluation, and saving arguments are controlled by this
      # submission wrapper and contain only fixed Miles CLI switches.
      # shellcheck disable=SC2086
      python3 scripts/run_deepseek_v4.py full-train \
        --model-name "${MODEL_NAME}" \
        --init-model-source "${INIT_MODEL_SOURCE}" \
        --rollout-weight-source "${ROLLOUT_WEIGHT_SOURCE}" \
        --num-nodes 8 \
        --num-gpus-per-node 4 \
        --hardware GB300 \
        --mode "${MODE}" \
        --run-id "${RUN_ID}" \
        --model-dir /workspace/data/models \
        --model-local-dir /workspace/data/models \
        --data-dir /workspace/data/datasets \
        --save-dir /workspace/data/runs \
        --megatron-path /root/Megatron-LM \
        ${OPTIMIZER_OFFLOAD_ARGS} \
        ${PRECISION_ARGS} \
        ${EVAL_ARGS} \
        ${R3_ARGS} \
        ${SAVE_ARGS} \
        ${FAULT_TOLERANCE_ARGS} \
        ${BACKEND_ARGS} \
        --extra-args "--allgather-cp ${REMATERIALIZE_ARGS} ${TRAIN_PRECISION_EXTRA_ARGS} ${LOG_PASSRATE_ARGS} ${WANDB_RUN_ID_ARGS} --offload-train-target ${OFFLOAD_TRAIN_TARGET} --offload-train-disk-dir ${OFFLOAD_TRAIN_DISK_DIR} --offload-train-disk-chunk-mb ${OFFLOAD_TRAIN_DISK_CHUNK_MB} --te-precision-config-file /workspace/data/configs/dsv4_te_precision.yaml --distributed-timeout-minutes 60 --disable-msc --http-request-timeout 300 --update-weight-buffer-size ${UPDATE_WEIGHT_BUFFER_SIZE} --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC} --num-rollout ${NUM_ROLLOUT} --use-wandb --wandb-project miles-run_deepseek_v4 --wandb-group ${RUN_ID} --disable-wandb-random-suffix ${EXTRA_ARGS_APPEND}"
    else
      while [[ ! -e "${ray_dir}/done" ]]; do
        sleep 5
      done
    fi
  '
