# DeepSeek-V4-Flash RL: Four-Step Validation

Date: 2026-08-11

## Result

Both configurations completed four GRPO rollout/train steps and four online
weight updates on 32 GB300 GPUs:

- BF16 training + FP8 rollout
- blockwise FP8 training + FP8 rollout

The online RL path passed. Neither run reported a CUDA IPC warning or CUDA OOM
after the producer/consumer ownership cleanup was applied.

The checkpoint path failed in both runs. Each final save wrote about 3.7 TiB of
`.distcp` shards, then checkpoint child processes exited with signal 11 and the
trainer actors remained blocked in `save_model`. The checkpoints had neither
`.metadata` nor `latest_checkpointed_iteration.txt`, so they were invalid. The
jobs were canceled after all four training steps, and the invalid shards were
deleted.

No longer run was submitted.

## Configuration

- Model: DeepSeek-V4-Flash-FP8
- Algorithm: GRPO
- Training data: DAPO Math 17k
- Evaluation: AIME-2024, 8 samples per problem
- Cluster: AGA, 8 nodes x 4 GB300 GPUs
- Trainer topology: TP2, PP8, CP2, EP4
- Rollout topology: one SGLang TP4/EP4 engine per node, 8 engines total
- Placement: colocated trainer and rollout
- Rollout batch: 32 prompts x 8 samples, maximum response length 4096
- Training steps: 4

The FP8 training arm used `--fp8-recipe blockwise`. It did not use MXFP8. Both
arms used the same FP8 rollout checkpoint and SGLang configuration.

## Accuracy and train/rollout mismatch

Initial AIME pass@1 was `0.441667` for the BF16-training arm and `0.450000` for
the FP8-training arm.

| Step | BF16 pass@1 | FP8 pass@1 | BF16 diagnostic KL | FP8 diagnostic KL | BF16 abs delta-logp | FP8 abs delta-logp |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.562500 | 0.589844 | 0.00241481 | 0.00334554 | 0.0225466 | 0.0266273 |
| 1 | 0.523438 | 0.484375 | 0.00237874 | 0.00329539 | 0.0224689 | 0.0259424 |
| 2 | 0.546875 | 0.546875 | 0.00234019 | 0.00329482 | 0.0215331 | 0.0261611 |
| 3 | 0.484375 | 0.515625 | 0.00218904 | 0.00321799 | 0.0202298 | 0.0254459 |
| Mean | 0.529297 | 0.534180 | 0.00233070 | 0.00328843 | 0.0216946 | 0.0260442 |

Over four steps, mean rollout pass@1 differed by 0.49 percentage points. This
segment is too short to establish a downstream accuracy difference.

The FP8-training arm had a consistently larger train/rollout mismatch. Its mean
diagnostic KL was 41% higher and its mean absolute log-probability difference
was 20% higher than BF16. Both metrics remained stable or decreased through
step 3; there was no divergence within this validation window.

## Timing observed during validation

| Metric | BF16 training | FP8 training |
|---|---:|---:|
| Cold train step | 1226.75 s | 1414.84 s |
| Mean steady train step, steps 1-3 | 153.79 s | 240.33 s |
| Mean FP8 rollout | 238.39 s | 239.86 s |

The rollout times match because both arms used the same FP8 inference path. The
FP8 training arm was slower in this configuration; performance was not the
acceptance criterion for this four-step run.

## Required development

### Online weight update

The four-step path requires the current Miles/SGLang ownership fixes:

- Miles performs a per-bucket rendezvous, releases producer references, and
  runs final CUDA IPC cleanup after the complete update.
- SGLang clones each received CUDA IPC bucket into SGLang-owned storage before
  loading it, then performs end-of-update garbage collection and CUDA IPC cache
  cleanup.

These changes belong in Miles and SGLang. Without them, repeated online updates
retain producer IPC allocations and eventually OOM.

### Checkpoint save

Checkpoint saving still requires development in the Megatron/PyTorch
distributed-checkpoint path. `async_save=False` and Miles' `--disable-msc` did
not prevent checkpoint subprocesses from being created, and those subprocesses
still exited with signal 11. A safe inline synchronous writer or a fix to the
remaining subprocess writer path is required before these runs can be resumed
from a saved checkpoint.

No additional training or SGLang kernel change was needed for the four-step
blockwise-FP8 training/rollout computation itself.

## Artifacts

- [BF16 training + FP8 rollout W&B](https://wandb.ai/megatron-core-moe-dev/miles-run_deepseek_v4/runs/agabf16gc20811105516)
- [FP8 training + FP8 rollout W&B](https://wandb.ai/megatron-core-moe-dev/miles-run_deepseek_v4/runs/agafp8gc30811112908)
- Detailed issue log: `experiments/aga_dsv4_flash/EXPERIMENT_LOG_2026-08-10.md`
- BF16 Slurm job: `422392`
- FP8 Slurm job: `422581`

