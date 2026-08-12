# DeepSeek-V4-Flash RL experiments: index

What to read, in the order it is worth reading. Every document here is a record
of a run that happened; none of them is a plan.

Talos links open the same file in the browser.

## Start here

| Document | Why |
|---|---|
| [dsv4_0731/STATUS.md](dsv4_0731/STATUS.md) · [talos](https://sc.talos.nvidia.com/view/home/scratch.lbo_gpu_1/projects/miles/experiments/dsv4_0731/STATUS.md) | The current experiment. Objective, checkpoint layout, conversion, run log, mismatch numbers, and what blocks each phase. |
| [dsv4_0731/WORKFLOW.md](dsv4_0731/WORKFLOW.md) · [talos](https://sc.talos.nvidia.com/view/home/scratch.lbo_gpu_1/projects/miles/experiments/dsv4_0731/WORKFLOW.md) | How the work is done: the three tiers, which copy of each package a job actually runs, how a numeric change is validated, how code reaches the cluster. |

## Current experiment — 0731 MXFP8 / MXFP4 rollout

Two phases against `deepseek-ai/DeepSeek-V4-Flash-0731`: MXFP8 train with MXFP8
rollout, then the same trainer against MXFP4 experts with MXFP8 activations, so
the mismatch delta isolates the rollout weight format.

Phase 1's mismatch numbers are in hand and reproduce across three runs. Phase 2
is at its first run that gets past the weight hand-over.

| File | Contents |
|---|---|
| [dsv4_0731/STATUS.md](dsv4_0731/STATUS.md) | Record of every job, including the three defects that blocked phase 2 and the evidence for each. |
| [dsv4_0731/patches/mxfp4_trtllm_hot_reload.patch](dsv4_0731/patches/mxfp4_trtllm_hot_reload.patch) | The SGLang change phase 2 runs with: keeps the loader attributes across a kernel-layout rebuild, and restores the load layout before weights arrive. |
| [dsv4_0731/patches/end_weight_update_sync.patch](dsv4_0731/patches/end_weight_update_sync.patch) | Kept only as a record. Measured harmful — the run it was meant to help got worse. |

Validation tools, in the order the workflow applies them:

| Script | Answers |
|---|---|
| [dsv4_0731/probe_mxfp4.py](dsv4_0731/probe_mxfp4.py) | What is actually in the release checkpoint's tensor headers. |
| [dsv4_0731/validate_mxfp4_dequant.py](dsv4_0731/validate_mxfp4_dequant.py) | Does our decode match SGLang's runtime decode on a real shard, bit for bit. |
| [dsv4_0731/make_mini_checkpoint.py](dsv4_0731/make_mini_checkpoint.py) | Slices a shard into a one-file checkpoint so the conversion entry point runs end to end. |
| [dsv4_0731/assert_fully_dequantized.py](dsv4_0731/assert_fully_dequantized.py) | Did any quantized payload or orphaned scale survive into the emitted artifact. |

Job scripts: [download_0731.sbatch](dsv4_0731/download_0731.sbatch),
[prepare_0731.sbatch](dsv4_0731/prepare_0731.sbatch), and the RL launcher
[aws_dsv4_flash/run_rl.sbatch](aws_dsv4_flash/run_rl.sbatch).

## Prior experiments

Read these for context on the platform, not for current conclusions — several of
their findings were superseded by the 0731 work.

| Document | Contents |
|---|---|
| [aga_dsv4_flash/HANDOVER_2026-08-11.md](aga_dsv4_flash/HANDOVER_2026-08-11.md) · [talos](https://sc.talos.nvidia.com/view/home/scratch.lbo_gpu_1/projects/miles/experiments/aga_dsv4_flash/HANDOVER_2026-08-11.md) | Entry point to the AGA bring-up: environment, credentials policy, what was left running. |
| [aga_dsv4_flash/FOUR_STEP_RESULTS_2026-08-11.md](aga_dsv4_flash/FOUR_STEP_RESULTS_2026-08-11.md) | The two-arm four-step result — FP8 train vs BF16 train, both with FP8 rollout. |
| [aga_dsv4_flash/BRINGUP_POSTMORTEM_AND_ACTION_PLAN_2026-08-11.md](aga_dsv4_flash/BRINGUP_POSTMORTEM_AND_ACTION_PLAN_2026-08-11.md) | What the bring-up cost and why. |
| [aga_dsv4_flash/EXPERIMENT_LOG_2026-08-10.md](aga_dsv4_flash/EXPERIMENT_LOG_2026-08-10.md) | Day-by-day log behind the postmortem. |
| [aws_dsv4_flash/README.md](aws_dsv4_flash/README.md) | The AWS GB300 two-arm smoke: what it runs and how to launch it. |
| [aws_dsv4_flash/RESULTS_2026-08-09.md](aws_dsv4_flash/RESULTS_2026-08-09.md) | Its result. |

## Live state

Neither of these is a file in the repo, and both change under you.

- **Metrics** — W&B project `megatron-core-moe-dev/miles-run_deepseek_v4`. Runs
  are grouped by `RUN_ID`; the mismatch metrics are `train/train_rollout_kl` and
  `train/train_rollout_logprob_abs_diff`.
- **Jobs** — on `oci-aga`, `sacct -j <id>` and
  `$WORKSPACE/logs/miles-dsv4-rl-<id>.log`, where the workspace is
  `/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl`.
  Each run log names the commit it was launched from; STATUS.md records the same
  SHA next to the job.
