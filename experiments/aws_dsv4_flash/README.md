# DeepSeek-V4-Flash RL on AWS GB300

This directory runs a controlled two-arm Miles smoke experiment:

1. blockwise FP8 training with FP8 rollout;
2. BF16 training with FP8 rollout.

Both arms use the same public FP8 reference checkpoint, datasets, GRPO
configuration, rollout settings, deterministic mode, R3, and 32-GPU colocated
topology. The only intended model-compute difference is the Megatron training
precision.

## Hardware and topology

- AWS Slurm cluster, 8 nodes x 4 NVIDIA GB300 GPUs (32 GPUs total)
- One NVL72 segment (`#SBATCH --segment=8`, `--switches=1`)
- Training: TP=2, PP=8, CP=2, SP enabled, EP=4, expert TP=1
- Rollout: eight SGLang engines, each TP=4, DP=1, EP=4
- Colocated rollout and training

The container is `radixark/miles:latest`. Pyxis must be launched with
`--no-container-mount-home`; otherwise the cluster's automatic home mount hides
the image's `/root/Megatron-LM` editable source tree.

## Data and checkpoints

Host root:

```text
/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl
```

- Rollout/reference checkpoint: `sgl-project/DeepSeek-V4-Flash-FP8`
- Training initialization: BF16 cast followed by distributed `torch_dist`
- Training data: `zhuzilin/dapo-math-17k`
- Evaluation data available: `zhuzilin/aime-2024`

`download.sbatch` downloads the Hugging Face assets on the CPU datamover
partition. `prepare.sbatch` performs the single-node FP8-to-BF16 cast. The first
32-GPU run performs the distributed BF16-to-`torch_dist` conversion through the
external Ray cluster; subsequent runs reuse its sentinel.

## Smoke comparison

The current smoke uses two rollout steps, the full recipe rollout batch of 32,
8 samples per prompt, and maximum response length 4096. Evaluation and
checkpoint saving are disabled for this pipeline validation. Fault tolerance is
also disabled when saving is disabled, matching the DeepSeek-V4 CI setup.

The precision switch is:

```text
FP8 arm:  --train-fp8    --rollout-fp8
BF16 arm: --no-train-fp8 --rollout-fp8, plus Megatron --bf16
```

`run_deepseek_v4.py` adds `--bf16` automatically only in its FP8/MXFP8
training branches. The launcher therefore passes `--bf16 --transformer-impl
transformer_engine` explicitly for the BF16 arm so that disabling FP8 changes
the training recipe to BF16 rather than leaving Megatron's dtype unspecified.

The 32-GPU preset uses context parallel size 2. DeepSeek-V4 rejects the default
zigzag context-parallel path, so the launcher also passes `--allgather-cp` for
both arms.

The full 256-sample rollout produces roughly 700K response tokens per step.
With PP=8, a valid pipeline send/receive can remain pending longer than Miles'
default 10-minute distributed timeout while later stages are still computing.
The launcher sets `--distributed-timeout-minutes 60`; this prevents the NCCL
watchdog from aborting active long-running stages.

The FP8 exception recipe must be visible to Ray actors on every node. The
upstream launcher materializes this YAML under the head node's `/tmp`, which is
not shared in a multi-node Pyxis job. This wrapper therefore passes the mounted
shared file explicitly:

```text
--te-precision-config-file /workspace/data/configs/dsv4_te_precision.yaml
```

The colocated weight update uses PyTorch CUDA IPC. In the current container,
the receiver removes an IPC shared-memory name before the producer's allocator
cleanup runs; PyTorch 2.11 then treats the second `shm_unlink` call's `ENOENT`
as fatal. `idempotent_shm_unlink.c` provides a narrow `LD_PRELOAD` compatibility
layer that converts only this already-absent result to success. Other unlink
results and errors are unchanged. The library is built for aarch64 on the AWS
login node and mounted into every Ray node.

The launcher defaults to the official 1 GiB weight-update buffer, but exposes
`UPDATE_WEIGHT_BUFFER_SIZE` for diagnosis. The r8 comparison uses 256 MiB for
both arms after two retries stalled inside CUDA synchronization during the
initial 1 GiB-bucket update. This changes transfer granularity only; model
precision, optimizer, data, and rollout settings remain identical.

The upstream deterministic recipe forces `NCCL_ALGO=Ring`. Three initial
weight updates stalled inside CUDA synchronization with Ring, including one
with the reduced buffer. A 32-rank NCCL smoke established that Tree cannot run
the required int8 all-gather, while Tree and NVLS cannot run the int8
broadcast. Explicit per-collective selection works in NCCL but Megatron rejects
it in deterministic mode. The AWS wrapper therefore uses Megatron's supported
`NCCL_ALGO=^NVLS`, which lets NCCL select a compatible deterministic algorithm
per collective while excluding NVLS. A 32-rank smoke passed both int8
all-gather and a 256 MiB int8 broadcast with this setting. Both precision arms
use the same selection.

The colocated updater launches asynchronous broadcasts on multiple NCCL
communicators. NCCL 2.28 leaves implicit cross-communicator launch ordering
disabled by default. The AWS wrapper enables `NCCL_LAUNCH_ORDER_IMPLICIT=1`,
which orders operations from different communicators by host launch order to
prevent deadlock. The helper passes this environment variable only when the
caller explicitly sets it, so other DeepSeek-V4 runs retain their existing
behavior. See the [NCCL 2.28 environment variable
documentation](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2283/user-guide/docs/env.html#nccl-launch-order-implicit).

The direct colocated CUDA IPC path also needs a bucket-level rendezvous. Each
rollout engine receives tensors gathered from four trainer ranks, but only the
group's source rank owns and waits for the engine RPC. Without a rendezvous,
the other ranks can release their IPC backing storage and launch the next
bucket's PP/EP/TP collectives while an engine is still importing the current
bucket. Miles now uses the existing trainer Gloo group to rendezvous after each
successful base-weight bucket and before releasing that storage. This keeps
the 32 ranks in one collective epoch.

Rendezvous alone did not fix the update hang because the routed-expert EP
broadcast selected an inconsistent root. When the original parameter owner was
outside the current EP group, every rank used itself as `src`; the members of
one process group therefore called the same broadcast with different roots.
Each parameter now records the owner's EP-local coordinate. Every EP group maps
that coordinate to one global rank with `dist.get_global_rank`, so all members
use the same root and that root already holds the parameter received through
the PP broadcast. Device synchronization between the PP, EP, and TP phases
also prevents separate NCCL communicators from overlapping phases.

The AWS GB300 wrapper disables optimizer CPU offload by default. With CPU
offload enabled, r14 consumed 5.42 TiB of host memory across eight nodes and
peaked at 829.7 GiB on one 900 GiB node before Slurm killed the job during
`actor_train`. At the same point, trainer GPU usage was about 66 GiB out of
276.6 GiB per GPU. Keeping optimizer states on GPU uses the available GB300
memory and avoids the host-memory limit. Set `OPTIMIZER_OFFLOAD=1` to restore
the upstream default.

The wrapper also enables `--rematerialize-param-from-master-weight` by default.
With the optimizer on GPU, the colocated trainer can discard its CPU parameter
backup during rollout and rebuild it from the distributed optimizer's master
weights before the next train step. This targets the remaining host-memory
peak during trainer sleep without changing model precision or optimizer math.
Set `REMATERIALIZE_PARAM_FROM_MASTER_WEIGHT=0` to disable it.

The synchronous driver now skips trainer offload and actor-to-rollout weight
sync after the final training step when no final evaluation is scheduled. No
consumer can use those weights: the loop exits immediately and disposes the
rollout engines. Avoiding that unnecessary final transition also prevents a
late host-memory peak in long-lived colocated actors.

DeepSeek-V4 freezes router parameters with `--moe-router-freeze-gate`. Frozen
router weights and routing tables are absent from Megatron's DDP param buffers,
so they have no distributed-optimizer master shard from which to rematerialize.
Miles now derives the extras backup from DDP buffer membership: every parameter
outside those buffers keeps a pinned backup, while buffered parameters are
rebuilt from optimizer masters. The AWS container test for this behavior passed
5/5 tests in job `2932835`.

Rollout HTTP requests use a 300-second timeout in this wrapper. Miles otherwise
constructs its shared `httpx` client with no timeout, so a response whose body
is not fully delivered can block the entire rollout indefinitely. The finite
timeout lets the existing retry loop recover such requests.

## W&B

W&B project: `miles-run_deepseek_v4`

Run groups:

- `dsv4-aws-fp8t-fp8r-smoke-r4-20260807`
- `dsv4-aws-fp8t-fp8r-smoke-r7-20260807`
- `dsv4-aws-bf16t-fp8r-smoke-r7-20260807`
- `dsv4-aws-fp8t-fp8r-smoke-r8-buf256m-20260807`
- `dsv4-aws-bf16t-fp8r-smoke-r8-buf256m-20260807`
- `dsv4-aws-fp8t-fp8r-smoke-r9-timeout-20260807`
- `dsv4-aws-bf16t-fp8r-smoke-r9-timeout-20260807`
- `dsv4-aws-bf16t-fp8r-smoke-r19-serialized-sync-20260809`
- `dsv4-aws-fp8t-fp8r-smoke-r20-ep-root-20260809`
- `dsv4-aws-bf16t-fp8r-smoke-r21-skip-final-sync-20260809`
- `dsv4-aws-fp8t-fp8r-smoke-r22-nvme-offload-20260809`
- `dsv4-aws-fp8t-fp8r-smoke-r23-nvme-offload-20260809`
- `dsv4-aws-fp8t-fp8r-smoke-r24-nvme-offload-finalize-20260809`
- `dsv4-aws-bf16t-fp8r-acc60-aime-20260810T040256Z`
- `dsv4-aws-fp8t-fp8r-acc60-aime-20260810T040256Z`
- `dsv4-aws-fp8rollout-aime-baseline-r2-20260810T072052Z`

The API key is never passed in the Miles command line because the default helper
would print it as part of `ray job submit`. Each node instead performs a quiet
`wandb login` into a job-local `/tmp` home before Ray starts. The key is unset
from the process environment, and the temporary credential directory is removed
at job cleanup. The source key remains only in the mode-600 shared secret file.

## Files

- `image_smoke.sbatch`: validates GB300, image dependencies, Miles CLI, and W&B
- `download.sbatch`: downloads model and datasets without occupying GPUs
- `prepare.sbatch`: casts the public FP8 checkpoint to BF16
- `dsv4_te_precision.yaml`: shared Transformer Engine precision exceptions
- `idempotent_shm_unlink.c`: CUDA IPC shared-memory cleanup compatibility layer
- `run_rl.sbatch`: starts the 8-node Ray cluster and runs either precision arm

## AWS job lineage

- `2892703`: rejected the default zigzag CP implementation; established the
  reusable 32-GPU `torch_dist` checkpoint before failing.
- `2893434`: initialized all eight FP8 SGLang engines, then exposed the
  head-node-local precision YAML bug when the Megatron actors started.
- `2893662`: completed one FP8 rollout, then hit the PyTorch CUDA IPC duplicate
  `shm_unlink` failure during the first training step; canceled after diagnosis.
- `2893663`: canceled dependency after `2893662` was stopped.
- `2894246`: crossed the CUDA IPC failure point, completed rollout and
  `compute_log_prob`, then hit the default 10-minute distributed watchdog while
  later PP stages were still computing. This established the need for the
  60-minute distributed timeout.
- `2894247`: canceled comparison dependency after `2894246` was stopped.
- `2895289`: first retry with the 60-minute timeout. All eight rollout engines
  and 32 trainer actors initialized, but the initial weight update made no
  progress after bucket 291/533 while two nodes remained in GPU collectives.
  All engine health checks returned 200 and GPU error counters were clean, so
  the run was canceled as a transient collective stall rather than waiting for
  the 60-minute watchdog.
- `2895290`: canceled comparison dependency after `2895289` was stopped.
- `2895811`: retried on a different NVL72 segment. All engines and trainer
  actors initialized, then the initial update stopped at bucket 113/533 for 10
  minutes. A `perf` sample placed rank 0 in `cuCtxSynchronize_v2`; all engine
  health checks returned 200 and no Python or CUDA error was logged.
- `2895812`: canceled comparison dependency after `2895811` was stopped.
- `2896588`: the 256 MiB weight-update buffer completed all 2130 buckets in
  54.2 seconds. SGLang then completed all 256 rollout requests within 90
  seconds, but the timeout-free Miles HTTP client remained blocked on 40
  router connections after collecting 216 samples; canceled after diagnosis.
- `2896589`: canceled comparison dependency after `2896588` was stopped.
- `2897194`: FP8 training and FP8 rollout retry with the 256 MiB buffer and a
  300-second rollout HTTP timeout. All actors initialized, but the initial
  Ring weight update stopped at bucket 895/2130 for more than five minutes;
  canceled after diagnosis.
- `2897195`: canceled comparison dependency after `2897194` was stopped.
- `2897772`: FP8 training and FP8 rollout retry using Tree collectives, the
  256 MiB update buffer, and the finite rollout HTTP timeout. SGLang initialized
  successfully, but Megatron initialization failed because Tree does not
  support the required int8 all-gather in NCCL 2.28.9.
- `2897773`: canceled comparison dependency after `2897772` failed.
- `2898462`: NCCL smoke showed that NVLS supports the int8 all-gather but not
  the 256 MiB int8 broadcast.
- `2898630`: NCCL smoke showed that Tree also does not support the int8
  broadcast.
- `2898825`: completed the 32-rank smoke with
  `allgather:NVLS;broadcast:Ring`.
- `2898912`: FP8 training and FP8 rollout retry with the smoke-validated
  per-collective NCCL selection. Megatron rejected the per-collective string in
  deterministic-mode argument validation before actor startup.
- `2898913`: canceled comparison dependency after `2898912` failed.
- `2899287`: completed the 32-rank smoke with Megatron's supported `^NVLS`
  setting; int8 all-gather and 256 MiB int8 broadcast both passed.
- `2899561`: FP8 training and FP8 rollout retry using the smoke-validated
  `^NVLS` deterministic configuration. It passed SGLang and Megatron
  initialization, then the 256 MiB update stopped at bucket 1794/2130 for more
  than five minutes.
- `2899562`: canceled comparison dependency after `2899561` was stopped.
- `2900481`: FP8 training and FP8 rollout retry with 64 MiB update buckets.
  The update advanced to bucket 7117/8537, then stopped for more than five
  minutes; this ruled out payload size as the sole cause.
- `2900482`: canceled comparison dependency after `2900481` was stopped.
- `2901142`: FP8 training and FP8 rollout retry with implicit
  cross-communicator NCCL launch ordering and 256 MiB update buckets. The
  initial update completed in 54.3 seconds, rollout collected all 256 samples
  in 69.6 seconds, and `compute_log_prob` completed in 588.1 seconds. During
  `actor_train`, CPU optimizer offload exhausted the 900 GiB host-memory limit
  on at least one node; Slurm reported 177 OOM kills after 44 minutes.
- `2901143`: canceled comparison dependency after `2901142` failed.
- `2915078`: FP8 training and FP8 rollout retry with optimizer states kept on
  GPU. It completed the initial weight update, all 256 rollout requests,
  `compute_log_prob`, and the first optimizer step. The step logged to W&B with
  `actor_train_time=873.8s` and `train_rollout_kl=0.00332`, then trainer sleep
  exhausted host memory while backing up model allocations; Slurm reported an
  OOM for step `2915078.0`.
- `2915079`: canceled comparison dependency after `2915078` failed.
- `2929617`: FP8 r16 used GPU optimizer state plus parameter rematerialization
  to remove the CPU backup that caused the r15 sleep-time OOM. It failed during
  trainer initialization because the original
  rematerialization coverage check could not restore frozen DeepSeek-V4 router
  parameters that are absent from DDP param buffers. W&B run: `02x6ba88`.
- `2929618`: BF16 r16 independently reproduced the same initialization failure.
  W&B run: `mtwwq3t6`.
- `2932835`: AWS Miles-container unit test for backing up parameters outside DDP
  buffers; all 5 tests passed.
- `2932839`: FP8 training and FP8 rollout r17 with the router rematerialization
  fix. All 32 trainers passed rematerialization coverage and slept without a
  host OOM, but the initial update stopped at bucket 1636/2130. An eight-second
  `perf` sample was saved before canceling the run; trainer ranks were active in
  CUDA driver/JIT paths rather than making bucket progress.
- `2932840`: BF16 training and FP8 rollout r17 with the same rematerialization
  fix and no dependency on the FP8 run. The update stopped at bucket 1531/2130;
  all four trainer ranks on one node were inside `cuCtxSynchronize_v2` while
  the other 28 ranks waited. The run was canceled after saving a `perf` sample.
- `2933704.1`: official Miles-container test step for the CUDA IPC rendezvous,
  rematerialization, and HTTP timeout changes; all 41 tests passed.
- `2933704`: FP8 training and FP8 rollout r18 with the per-bucket CUDA IPC
  rendezvous and explicit CUDA synchronization between collective phases. The
  update stopped at bucket 1421/2130. All four ranks on one node were again in
  `cuCtxSynchronize_v2` while the other 28 waited. Code inspection then found
  that routed-expert broadcasts used a different root on every rank whenever
  the original owner was outside the current EP group.
- `2933937`: BF16 training and FP8 rollout r19 with serialized collective
  phases, per-bucket rendezvous, and the corrected EP-broadcast root. Its first
  2130-bucket update completed on all 32 ranks in 69.5--71.4 seconds, crossing
  every bucket where earlier runs stopped. Its first 256-sample rollout took
  66.9 seconds and its first `compute_log_prob` took 451.9 seconds. Both
  optimizer steps completed and were uploaded to W&B, but the driver then
  performed an unnecessary final trainer offload and weight update. The head
  node reported host OOM during that final transition; Slurm recorded a
  625,937,280 KiB step-task peak RSS, with colocated Ray/SGLang processes and
  pinned backups sharing the same node. W&B run: `ijtpqxvn`.
- `2934076`: FP8 training and FP8 rollout r20 with the same fixes. Submitted
  independently after the corrected EP-root implementation and its container
  unit test passed. Its first update completed on all 32 ranks in 70.3--70.4
  seconds, its first 256-sample rollout took 71.9 seconds, and its first
  `compute_log_prob` took 589.6 seconds. The FP8 optimizer step completed in
  866.8 seconds with `train_rollout_kl=0.00338`. Its first post-step trainer
  offload then exhausted host memory; Slurm recorded a 687,489,408 KiB peak RSS
  on one task before the node OOM. W&B run: `srxjbhrz`.
- `2934638`: BF16 training and FP8 rollout r21 with the final-transition skip.
  Completed both rollout/training steps and exited 0. The final-transition skip
  prevented the r19 end-of-run host OOM. The missing final trainer history row
  was appended from the completed actor log and verified through the W&B API.
  W&B run: `e1s53krn`.
- `2934639`: canceled before allocation because r20 established that the FP8
  actor's post-step CPU backup does not fit alongside colocated SGLang.
- `2934761`: FP8 training and FP8 rollout r22 with the final-transition skip
  plus Miles' disk-backed trainer offload. It exited during CLI parsing because
  the model launcher does not expose `--offload-train-target` as a top-level
  option; no model actors or W&B run were created.
- `2934815`: r23 moves the same disk-offload switches into the launcher's
  `--extra-args` passthrough. Argument validation correctly rejected combining
  disk offload with parameter rematerialization; no model actors or W&B run were
  created.
- `2935340`: r24 uses disk offload without parameter rematerialization. It also
  explicitly finalizes trainer and rollout tracking before Ray teardown so the
  last trainer step is uploaded to W&B. Phase-boundary actor backups use the
  27.3 TiB node-local NVMe RAID at `/raid/scratch`; optimizer state remains on
  GPU during training because the completed r20 step showed that it fits. Both
  rollout/training steps completed, the post-step disk offload crossed the r20
  OOM point, the job exited 0, and W&B history contains both trainer steps.
  W&B run: `cumrxk9s`.
- `2941595` and `2941596`: initial 16-hour BF16 and FP8 accuracy submissions.
  Neither allocated. Slurm estimated that a 16-hour segment would not start
  until August 12, so both were canceled and replaced by checkpointed 10-step
  chunks.
- `2943854`: first eval-only AIME baseline attempt. It exposed that the
  Megatron optimizer scheduler rejected `num_rollout=0`, despite `train.py`
  having a dedicated eval-only path. No evaluation was run.
- `2944166`: eval-only retry after giving optimizer initialization a one-step
  scheduler horizon. It completed in 18 minutes 46 seconds and uploaded the
  initial FP8-rollout AIME point to W&B run `q4n7f2bb`: pass@1 0.43333,
  pass@2 0.50119, pass@4 0.53000, and pass@8 0.53333.
- `2944229`: official Miles-container regression test for the eval-only
  scheduler fix; both tests passed.
- `2944532`: first BF16-training accuracy chunk, targeting rollouts 0 through
  9. It writes checkpoints under the original 60-step run directory and uses
  fixed W&B run ID `550d0fef` so later chunks extend one curve.
- `2944533`: first FP8-training accuracy chunk with the same target range and
  fixed W&B run ID `80ef8b50`. It retains the validated NVMe trainer-offload
  configuration.
- `2944605`, `2944607`, `2944609`, `2944611`, and `2944613`: BF16-training
  continuation chunks targeting cumulative rollout counts 20, 30, 40, 50,
  and 60. Each job depends only on the preceding BF16 chunk.
- `2944606`, `2944608`, `2944610`, `2944612`, and `2944614`: FP8-training
  continuation chunks targeting the same cumulative rollout counts. Each job
  depends only on the preceding FP8 chunk, so the two precision arms schedule
  and recover independently.
- The two accuracy chains above were canceled at 04:11 PDT on August 10 while
  every job was still pending. They allocated no GPUs and produced no new
  checkpoints or W&B history.
