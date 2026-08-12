#!/bin/bash
#SBATCH --job-name=dsv4-nccl-smoke
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl/logs/%x-%j.log
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --segment=8
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --switches=1

set -euo pipefail

code=/home/lbo/work/dsv4_flash_rl/miles
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)
export MASTER_PORT=29641
export NCCL_ALGO=${NCCL_ALGO:-NVLS}
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

srun --kill-on-bad-exit=1 \
  --container-image=radixark/miles:latest \
  --no-container-mount-home \
  --container-mounts="${code}:/workspace/miles" \
  --container-workdir=/workspace/miles \
  bash -lc '
    export RANK=${SLURM_PROCID}
    export WORLD_SIZE=${SLURM_NTASKS}
    export LOCAL_RANK=${SLURM_LOCALID}
    python3 experiments/aws_dsv4_flash/nccl_algo_smoke.py
  '
