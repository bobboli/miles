# DeepSeek-V4-Flash RL on AGA: Experiment Log

Date: 2026-08-10

## Scope

- Cluster: `oci-aga-slurm-1-login-01.nvidia.com`
- Slurm account: `coreai_devtech_all`
- Hardware: 8 nodes x 4 NVIDIA GB300 GPUs in one segment
- Algorithm/data: GRPO on DAPO Math 17k, AIME-2024 evaluation
- Comparison: BF16 train + FP8 rollout versus FP8 train + FP8 rollout
- Weight-update bucket size: 64 MiB for BF16 training and 256 MiB for FP8
  training, selected independently to preserve transfer headroom without adding
  unnecessary collective epochs

## Why the first day produced diagnostics rather than an accuracy curve

- About seven hours were lost waiting for the original 60-step jobs
  (`416113` and `416191`) in the long-QoS queue. They never started and were
  replaced by checkpointed four-step segments under the short QoS.
- The successful two-step smokes did not execute the third or fourth online
  weight update. Multi-step CUDA IPC/memory failures therefore appeared only
  after the smoke gate had passed; both formal arms had to restart from step 0
  because neither failed run produced a checkpoint.
- Two launches (`420258` and `420259`) were wasted by an invalid wrapper value,
  `MODE=full`, instead of the supported `MODE=normal`.
- The wrapper was then changed from the AWS-validated GPU optimizer setting
  (`OPTIMIZER_OFFLOAD=0`) back to the upstream CPU-offload default. That was a
  configuration regression: the AWS runbook had already established that four
  trainer ranks plus colocated SGLang exceed host RAM with CPU optimizer
  offload. Jobs `420997` and `420998` reproduced that host OOM near the end of
  train step 0 and produced no checkpoint.
- The real implementation issue was the CUDA IPC tensor lifetime in the online
  weight-update path. SGLang device synchronization plus Python reference
  deletion was insufficient. The current fix clones each flattened bucket into
  SGLang-owned CUDA storage before model loading, then synchronizes and releases
  the trainer-owned IPC tensor.
- Jobs `420997` and `420998` proved the ownership fix at the complete initial
  2,130-bucket boundary, but CPU optimizer offload stopped both during step 0.
  The replacement pair returned optimizer state to GPU: BF16 used parameter
  rematerialization, while FP8 used the already validated NVMe trainer-offload
  path. Both replacements also failed before training. BF16 exhausted GPU
  headroom during the initial update; FP8 completed that update but exposed a
  roughly 164 GiB retention anomaly on the head-node reporting GPU before
  SGLang resume. Until both cross
  the later updates and save checkpoints, there is no formal FP8-versus-BF16
  curve.

## Issues and findings

### Colocated online weight update can stop making progress

The first AGA attempts used the launcher's 1 GiB default bucket size. FP8 job
`414005` stopped during the initial online update at 41/533 buckets. It was
canceled rather than waiting for the 60-minute distributed timeout. This
matches the earlier AWS observation that the 1 GiB path is not reliable for
this topology.

The corrected 256 MiB runs also exposed a remaining intermittent collective
ordering problem:

- FP8 job `414203` stopped at 1847/2130 buckets.
- BF16 job `414204` reached 2130/2130 on the progress-reporting rank, but not
  all trainer ranks completed the update/finalization.
- In each job, the four trainer ranks on one pipeline-stage node stayed near
  100% GPU utilization while the other stages waited.

A separate 32-rank test, job `414378`, completed an object all-gather and a
256 MiB uint8 broadcast with `NCCL_ALGO=^NVLS`. Basic cluster-wide NCCL is
therefore working; the failure is specific to the PP/EP/TP communicator
sequence used by the colocated weight updater.

Working diagnosis: CUDA completion is synchronized locally between PP, EP,
and TP phases, but ranks do not rendezvous globally before switching to a
communicator with a different rank set. NCCL implicit launch ordering still
requires a consistent host launch order across devices.

Diagnostic FP8 job `414483` subsequently completed both rollout/train steps.
Its initial 2,130-bucket online update finished on all 32 trainer ranks in
73.4 seconds, and the second update finished in 63.3 seconds. This establishes
that the 256 MiB configuration can complete on AGA, but one successful run is
not enough to call the earlier intermittent ordering issue fixed.

The direct weight iterator now adds a trainer-wide Gloo rendezvous after the
PP and EP collective phases. This prevents ranks from entering the next
communicator topology at different times. The focused unit test passed 7/7 in
the Miles container. Jobs `414842` and `414843` are the first end-to-end FP8
and BF16 smokes using this change.

### NCCL trace override must occur after container login setup

Diagnostic job `414410` was submitted with `NCCL_DEBUG=TRACE`, but the
container login environment reset it to `WARN`. The job was canceled before
model initialization completed. `run_rl.sbatch` now supports optional
post-profile overrides through `MILES_NCCL_DEBUG`,
`MILES_NCCL_DEBUG_SUBSYS`, and `MILES_NCCL_LAUNCH_RACE_FATAL`; normal runs are
unchanged when these variables are absent.

Replacement job `414483` ran with `NCCL_DEBUG=TRACE` and `COLL` trace output.
Its run ID was
`dsv4-aga-fp8t-fp8r-diag-trace-20260810T163847Z`; it completed successfully
with Slurm exit code 0.

### Explicit sbatch exports must not hide wrapper-computed variables

The first SGLang-owned-bucket submissions, jobs `420965` and `420966`, were
submitted with an explicit `--export` list that omitted the `ALL,` prefix.
The batch wrapper computed `HEAD_NODE`, but Slurm did not propagate that new
variable into the nested `srun`; tasks exited with `HEAD_NODE: unbound variable`
before Ray or the model started. FP8 was canceled once the shared submission
error was identified. The replacement submission includes `--export=ALL,...`,
and the wrapper now sets `SLURM_EXPORT_ENV=ALL` before `srun` so later computed
variables are propagated even if a caller supplies an explicit export list.

### CPU optimizer offload exceeds per-node host memory

Jobs `420997` and `420998` used `--optimizer-cpu-offload`. Both completed the
initial online update and the numerical work for train step 0, then Ray killed
a trainer when aggregate node memory crossed its 95% protection threshold.
The FP8 arm reached 880.22 GiB of 920.00 GiB reported usable memory; its four
trainer processes used 192.11--200.48 GiB each. The BF16 arm reached 878.44
GiB; its four trainers used 177.62--189.67 GiB each. Colocated SGLang scheduler
processes added roughly 22--23 GiB per GPU. Neither failure was a CUDA OOM or
an online-update failure.

This repeats the AWS finding already recorded in this repository: on GB300,
the optimizer state must remain on GPU. The prior interpretation that jobs
`420319` and `420320` proved GPU optimizer residency invalid was too strong;
those jobs still used the leaky IPC transfer path. The corrected retry combines
GPU optimizer state with the SGLang-owned bucket clone. BF16 retains parameter
rematerialization; FP8 disables rematerialization and uses node-local NVMe for
trainer sleep, matching the previously successful AWS configurations. Raising
Ray's kill threshold would only hide the capacity error and is not the chosen
fix.

### GPU optimizer retry exposes a head-node retention anomaly

Jobs `421318` and `421319` restored the AWS-validated GPU-optimizer posture and
used the SGLang-owned flattened-bucket clone. Both were clean new W&B runs, but
neither reached evaluation or training.

BF16 job `421319` failed at approximately bucket 1,644 of 2,130 during the
initial online update. The next 130 MiB flattened-bucket allocation failed with
only 109--129 MiB free. The trainer occupied about 201.56 GiB and colocated
SGLang about 74.93 GiB. This is a per-GPU transfer-headroom failure; the run did
not emit the more-than-1,000-live-IPC-block warning.

FP8 job `421318` completed all 2,130 buckets in 73.8 seconds. Trainer rank 0 on
the head node ended with only 27.23 GiB free and 227.05/234.51 GiB
allocated/reserved, approximately 164 GiB less free than the representative
remote rank. Ray collapsed the other 31 memory lines into one deduplicated
sample at roughly 191.4 GiB free, so the aggregate log does not prove that only
rank 0 retained memory. The TP4 SGLang engine on the head node then failed
`cuMemCreate` on all four GPUs while the other seven engines resumed.

The producer cleanup ran `empty_cache()` before `ipc_collect()`. That cannot
return blocks which are still protected by CUDA IPC references; after
`ipc_collect()` makes them inactive, no second `empty_cache()` returned them to
the CUDA driver. The next retry explicitly deletes per-bucket references, runs
Python GC, calls `ipc_collect()`, and only then calls `empty_cache()`. Saving and
fault tolerance were also both enabled in `421318`, unlike the AWS smoke. The
retry keeps checkpoint saving but disables fault tolerance so the two behaviors
are no longer coupled.

Neither job saved a checkpoint. The next controlled pair must remove the rank-0
retention and use a smaller online-update bucket for BF16 before it can produce
an accuracy curve.

### A 64 MiB bucket fixes BF16 headroom but is too collective-heavy for FP8

The corrected-cleanup pair used 64 MiB buckets, producing 8,537 updates instead
of 2,130. BF16 job `421560` completed all 8,537 buckets in 160 seconds, returned
to 178.31 GiB free on rank 0, and resumed all eight SGLang engines. This crosses
the previous BF16 failure boundary and validates the smaller bucket for that
arm. Its initial AIME-2024 evaluation completed all 240 samples with
pass@1/pass@2/pass@4/pass@8 of
`0.4667`/`0.5143`/`0.5324`/`0.5333`. Rollout 0 then completed all 256 samples
in 248.1 seconds, with pass@1/pass@2/pass@4/pass@8 of
`0.5703`/`0.7701`/`0.9080`/`1.0000`. Train step 0 completed in 1,235.1
seconds (`compute_log_prob` 437.6 seconds and `actor_train` 796.9 seconds),
with diagnostic train/rollout KL `0.00221568` and mean absolute log-probability
difference `0.02092499`. Its second 8,537-bucket update then completed in
150.6 seconds, left rank 0 with 176.80 GiB free, and resumed SGLang for
rollout 1. This crosses the post-step resume boundary that failed in earlier
BF16 attempts. Rollout 1 produced pass@1/pass@2/pass@4/pass@8 of
`0.5430`/`0.7366`/`0.8826`/`1.0000`. Train step 1 took 156.0 seconds and
reported KL `0.00238895` and mean absolute log-probability difference
`0.02203244`. The third update completed all 8,537 buckets in 149.8 seconds
with 176.79 GiB free on rank 0. This is well beyond the third-update bucket
746 failure in job `419777`, with stable post-update memory. Rollout 2 had
pass@1 `0.5117`, and train step 2 took 153.1 seconds with KL `0.00235159`.
The fourth update completed in 151.8 seconds with 176.78 GiB free, then resumed
all rollout GPUs. Rollout 3 had pass@1 `0.5156`; final train step 3 and the
checkpoint save remain.

FP8 job `421559` made steady progress to 8,009/8,537, then stopped producing
logs. After almost three minutes without a new bucket, all 32 GPUs remained
allocated and nearly all reported 100% utilization, matching the earlier
collective-stall signature rather than an OOM. The job was canceled rather than
waiting for the 60-minute distributed timeout. FP8 does not need the smaller
allocation for headroom, so replacement job `421770` returns that arm to the
previously successful 256 MiB bucket size while retaining corrected IPC cleanup.
The replacement completed all 2,130 initial buckets in 73.0 seconds. Rank 0
returned to 191.75 GiB free, all eight SGLang engines resumed with HTTP 200,
and the 64 MiB run's collective stall did not recur at this boundary. Its
initial AIME-2024 pass@1/pass@2/pass@4/pass@8 values were
`0.4250`/`0.4964`/`0.5324`/`0.5667`, and rollout 0 produced
`0.5586`/`0.7411`/`0.8701`/`1.0000`. Train step 0 took 1,401.4 seconds
(`compute_log_prob` 562.7 seconds and `actor_train` 838.5 seconds), with KL
`0.00354038` and mean absolute log-probability difference `0.02842318`. Its
second update completed in 66.3 seconds with 189.65 GiB free on rank 0. Later
updates and the segment checkpoint remain the acceptance gates.

### The trace run's long first step was trainer warm-up, not a rollout hang

Job `414483` produced all 256 responses in 67.63 seconds in rollout 0 and
68.44 seconds in rollout 1. While the first training step was still executing,
the SGLang metrics correctly showed no running or queued requests and the
RolloutManager's router connections were idle. That state was initially
mistaken for response-collection blockage.

The actual long tail was trainer compute: the first `train` call took
1,407.3 seconds (`compute_log_prob` 566.7 seconds and `actor_train` 843.1
seconds), while the second took 228.7 seconds (92.8 and 144.5 seconds). The
first-step cost includes trainer warm-up and extremely verbose NCCL collective
tracing. It should not be used as a steady-state performance number.

### Phase rendezvous needs more colocated GPU headroom

The first paired runs with the phase-rendezvous change, FP8 job `414842` and
BF16 job `414843`, both reached the initial online update and made steady
progress, then failed with a CUDA OOM while constructing a 130 MiB flattened
weight bucket. FP8 reached bucket 1,660/2,130; BF16 reached 1,823/2,130.

At failure, SGLang occupied about 201.8 GiB of each 276.6 GiB GPU and only
49--105 MiB remained free. The default 0.7 static-memory fraction had created
a 16,560,640-token KV pool per engine, far above this experiment's demand.
The failure is therefore insufficient colocated transfer headroom, not a
collective stall. The next retry lowers `--sglang-mem-fraction-static` to 0.6;
the launcher exposes this as `SGLANG_MEM_FRACTION_STATIC` so the default recipe
remains unchanged.

### Producer-side CUDA IPC memory must be reclaimed before rollout resume

FP8 retry `415036` completed all 2,130 initial weight-update buckets on all 32
trainer ranks in 73.6 seconds, confirming that the phase rendezvous and 0.6
SGLang fraction provide enough transfer headroom. The job then failed while
resuming KV-cache and CUDA-graph memory: seven SGLang engines returned HTTP 200,
while the engine on `10.49.70.253` failed `cuMemCreate` with
`CUDA_ERROR_OUT_OF_MEMORY` and exited.

This is not explained by the static-memory fraction alone. Immediately after
the update, trainer logs reported about 191 GiB free per GPU, and successful
FP8 job `414483` resumed a larger 0.7 SGLang allocation with comparable free
memory. The base tensor-transfer path released each Python bucket after its
consumer completed, but did not explicitly collect producer-side CUDA IPC
storage or empty the caching allocator before the driver resumed SGLang.

The retry adds `clear_memory()` followed by `torch.cuda.ipc_collect()` after
all engines finish the update and all trainer ranks rendezvous. FP8 smoke
`415487` and BF16 smoke `415660` are the first end-to-end tests of that cleanup.
They keep the SGLang fraction at 0.6, so the cleanup is the only material
memory-lifecycle change.

FP8 smoke `415487` completed both steps with Slurm exit code 0. The initial
and post-train updates completed in 72.6 and 68.0 seconds, respectively, all
eight SGLang engines resumed after each update, and both rollout/train pairs
finished. Its two diagnostic train/rollout KL values were `0.00334564` and
`0.00338576`. This validates the cleanup for the FP8 path at both previously
failing lifecycle boundaries. W&B emitted `BrokenPipeError` messages from
atexit callbacks after the run had synced, but Ray reported the job succeeded;
these are teardown noise rather than an experiment failure.

BF16 smoke `415660` also completed both steps with Slurm exit code 0. Its
initial and post-train updates took 76.4 and 65.4 seconds, respectively, and
its diagnostic train/rollout KL values were `0.00231990` and `0.00234845`.
The two-step means are `0.00336570` for FP8 training and `0.00233418` for BF16
training, making the FP8 smoke mean about 44% higher. This is an early mismatch
signal, not an accuracy conclusion; the paired 60-step curves are required.

Both smoke runs are marked `finished` in W&B, and their history can be read
through the W&B API. The stored histories include rollout reward and response
statistics as well as `train/train_rollout_kl` and
`train/train_rollout_logprob_abs_diff`; no post-processing or log re-upload is
needed before comparing the full runs.

### A two-step smoke does not exercise the third online weight update

The first segmented BF16 accuracy run, job `419777`, completed train steps 0
and 1, then failed during the third online weight update at bucket 746/2,130.
The last-pipeline-stage trainer ranks 29 and 30 could not allocate the next
130 MiB flattened bucket; each GPU had only 81 MiB free. The error attributed
about 76.08 GiB to its colocated SGLang process and 200.44 GiB to its trainer
process. Shortly before the OOM, PyTorch warned that the producer had tried to
deallocate more than 1,000 CUDA IPC blocks that were still referenced by
consumer processes.

The earlier two-step smokes did not cover this boundary. With
`--num-rollout 2`, the loop performs the initial update, train step 0, one
post-train update, and train step 1, then exits. A third update only occurs
when a run proceeds to step 2. Producer-side `torch.cuda.ipc_collect()` was
therefore sufficient for the tested two-update lifecycle but is not a complete
multi-step fix for the BF16 arm.

The current evidence points to CUDA IPC references or asynchronous copies on
the SGLang consumer remaining live after an update response. The next fix must
retire the consumer CUDA work and release its deserialized tensor references
before the producer drops a bucket. Lowering the KV-cache allocation again
would only mask that lifetime error and would not explain the explicit CUDA
IPC warning.

FP8 job `419778` confirmed that this is a shared multi-step lifecycle bug, not
a BF16-only failure. FP8 completed the third update in 65 seconds and train
step 2, but then emitted the same more-than-1,000-live-IPC-block warning and
OOMed while constructing a 130 MiB bucket during the fourth update. The
progress-reporting rank had reached approximately 864/2,130 buckets. Its last
pipeline-stage GPUs had about 109 MiB free, with approximately 76.03 GiB used
by SGLang and 200.46 GiB used by the trainer. FP8 survived one additional
update because its earlier memory pressure was lower; it did not reclaim the
consumer-held blocks correctly either.

Job `419777` has valid diagnostic metrics for steps 0 and 1, including
train/rollout KL of `0.00224235` and `0.00234750`, but it did not reach the
segment checkpoint. Its W&B run must be treated as a failed diagnostic, and
the BF16 accuracy arm must restart from step 0 after the consumer-side fix is
validated beyond two steps. FP8 job `419778` likewise has valid diagnostic
metrics through step 2, with train/rollout KL values `0.00331197`,
`0.00322642`, and `0.00324599`, but it also failed before the segment
checkpoint. Both controlled accuracy arms must therefore restart cleanly.

Inspection of SGLang commit
`cb05a44f35a7c9e27e46d74112cc841ca674ef43` found the missing lifetime
boundary in
`SchedulerWeightUpdaterManager.update_weights_from_tensor`. The loader queues
GPU work, then the scheduler immediately executes a CPU-process-group barrier
and returns the HTTP response. It does not synchronize the device or explicitly
release the deserialized IPC tensors. DeepSeek-V4's GPU tensor load path does
not use its CPU-only asynchronous-loading thread pool, so Python can enqueue
many buckets faster than CUDA retires their source storage.

The proposed SGLang production change is two lines: synchronize the current
device after the selected runners finish loading, then delete
`named_tensors` before the CPU barrier and response. The experiment launcher
can apply this exact patch at container startup through `SGLANG_PATCH_FILE`;
the patch is syntax-checked and fails the job immediately if it no longer
applies to the image. The next paired four-step jobs are the end-to-end
validation: both must cross the fourth update and save a checkpoint before the
60-step curves restart.

Jobs `420258` and `420259` started together under the short QoS. Their Slurm
logs show the patch applying successfully in all eight containers per arm, and
read-back through each head node's container mount confirmed that the running
SGLang source had the synchronize-and-delete sequence in the intended order.
Both jobs then exited before model launch because the submission set
`MODE=full`; the wrapper's complete recipe mode is named `normal`, while its
only other mode is `debug_minimal`. This was a submission error, not an RL or
patch failure. Replacement jobs `420319` and `420320` use `MODE=normal`, clean
run/W&B IDs, and otherwise identical settings.

An upstream overlap check found no existing fix for this exact path. Open
SGLang PR [#17870](https://github.com/sgl-project/sglang/pull/17870) addresses
an MTP-specific leak where draft and target workers open the same
`LocalSerializedTensor` IPC handle twice. These DeepSeek-V4 runs have no draft
worker and send a flattened CUDA tensor directly, so that change does not cover
the observed leak. Closed issue
[#8076](https://github.com/sgl-project/sglang/issues/8076) reports the broader
`update_weights_from_tensor` OOM symptom but was closed for inactivity without
a fix. SGLang main at commit `585c3c6816b6ec8d5476eef44a0b5e8beeb3e5e3`
still returns from the tensor update after only a CPU-process-group barrier;
it has no device synchronization for the transferred tensor lifetime.

The replacement jobs completed their initial patched online updates in 73.5
seconds for FP8 and 74.4 seconds for BF16, with no CUDA IPC warning or OOM.
This is effectively the same initial-update time as the unpatched attempts;
the new device synchronization did not create a visible transfer slowdown.
Their initial AIME-2024 pass@1/pass@8 values were `0.4458`/`0.5667` for FP8
training and `0.4542`/`0.5667` for BF16 training. Both then entered rollout 0.
The decisive validation remains the later third and fourth updates, where the
unpatched jobs accumulated enough outstanding IPC storage to fail.

BF16 replacement job `420320` completed train step 0 and its full second
2,130-bucket update in 64.9 seconds, so the SGLang consumer synchronization
patch did retire the transfer work and did not reproduce the old bucket-level
failure. The run then failed while resuming one SGLang engine's KV-cache and
CUDA-graph pool. The failing trainer rank reported only 57.63 GiB free after
the update, and the engine logged `cuMemCreate CUDA_ERROR_OUT_OF_MEMORY`; the
other seven engines resumed successfully.

After the first optimizer step, the BF16 path rematerialized its approximately
118 GiB parameter buffer for weight export. At SGLang fraction 0.6, the rollout
engine needed approximately 98 GiB to restore its KV/graph pool, but the
reporting GPU had only 57.63 GiB free. At the time this was attributed to GPU
optimizer residency. That conclusion was incomplete: this run still used the
consumer-reference-leaking IPC path, while the existing AWS runbook had already
validated GPU optimizer state plus parameter rematerialization on GB300. Job
`420320` has one diagnostic training point in W&B but no checkpoint.

FP8 replacement job `420319` then completed step 1 but failed during the
third online update. At approximately bucket 1,120, trainer ranks 1 and 2 on
the reporting node could not allocate the next 130 MiB flattened bucket. Each
trainer process occupied about 200.55 GiB, its colocated SGLang process about
76.01 GiB, and only 37 MiB remained free. The PyTorch warning about more than
1,000 producer CUDA IPC blocks still referenced by consumers appeared despite
the SGLang synchronize-and-delete patch.

This evidence established that SGLang device synchronization alone did not
eliminate the within-update IPC reference backlog. The subsequent decision to
restore CPU optimizer offload on both arms was useful for separating the
initial-transfer failure from post-step GPU residency, but it was not a viable
long-run configuration on these roughly 1 TiB nodes.

The recipe-default retry established that the IPC-lifecycle fix is mandatory.
FP8 job `420703` failed during its first online update at bucket 1,791/2,130;
BF16 job `420704` failed at bucket 1,731/2,130. Both emitted the warning that
more than 1,000 CUDA IPC blocks were still referenced by consumer processes,
then failed to allocate the next 130 MiB flattened bucket with only 53--91 MiB
free. The failing GPUs attributed about 74.56 GiB to SGLang and about 201.96
GiB to the trainer process.

These failures occurred before the first optimizer step, so optimizer placement
could not address this boundary. The SGLang device synchronization and
`del named_tensors` patch was therefore insufficient: the deserialized
flattened bucket remained consumer-referenced after the HTTP response while its
storage was still owned by the trainer. The next controlled change cloned each
incoming flattened bucket into SGLang-owned CUDA storage before loading it and
released the original IPC-backed tensor immediately. This explicitly severs
producer storage lifetime from model-loader references; the experiment must
complete all 2,130 initial buckets without the IPC warning, then cross four
update boundaries and save a checkpoint.

Corrected jobs `420997` and `420998` have now passed the first of those
boundaries. The FP8-training arm completed all 2,130 initial buckets in 72.4
seconds and the BF16-training arm in 66.8 seconds. Neither emitted the
more-than-1,000-live-IPC-block warning or an OOM, and all eight SGLang engines
resumed in both jobs. Representative post-update free memory was 191.75 GiB in
the FP8 arm and 177.69--178.00 GiB in the BF16 arm. This is strong evidence that
the SGLang-owned clone fixes the initial-transfer leak, but it is not yet the
multi-step acceptance result: the second through fourth updates and checkpoint
save remain mandatory.

Their initial AIME-2024 results are also online in W&B. FP8 training produced
pass@1/pass@2/pass@4/pass@8 of `0.4667`/`0.5357`/`0.5967`/`0.6667`; BF16
training produced `0.4750`/`0.5214`/`0.5590`/`0.6000`. These are baseline
evaluation points before training, not evidence for either training precision.
Both jobs later hit the CPU-optimizer host-memory failure described above near
the end of step 0. Their step-0 diagnostic KL values were `0.00351571` for FP8
training and `0.00236877` for BF16 training; neither produced a checkpoint.

### Reproducibility fingerprints

- DAPO Math 17k training file: 17,398 rows, 10,490,834 bytes, SHA-256
  `cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6`.
- AIME-2024 evaluation file: 30 rows, 13,029 bytes, SHA-256
  `5d4685dd504391b0ec00a9b73c379a80ba4da804829a113b1d4fe630308d47cc`.
- `DeepSeek-V4-Flash-FP8/config.json`: SHA-256
  `52b5a1aa87606cb5be4f3158d706594edb1c4ce97ce6b1cd6079f15df075d7f5`.
- Trainer Engine precision YAML: SHA-256
  `7b419ce73735b251c0cc42479015bf9d8ffb20857a5795bac9db290ec7efa4c9`.
- Miles checkout base commit: `862ac1ea1fba864171b006d45e0d5e92ff008c6a`;
  the experiment also uses the tracked working-tree changes listed in this log
  and the eventual report/PR.
- Runtime image: `radixark/miles:latest`. Pyxis logged the tag but not a content
  digest, so the exact image digest is currently a reproducibility gap. Future
  recipe runs should submit a digest-pinned image or record the imported image
  digest explicitly.

BF16 job `415038`, which did not include the cleanup, provides a second reason
for it. The initial update completed in 76.5 seconds, rollout 0 completed in
67.9 seconds, and the first training step completed in 1,228.9 seconds. During
the next update, producer-side memory accumulated until the run failed at
bucket 1,025/2,130 while allocating a 130 MiB atomic bucket. The failing GPUs
had only 87--127 MiB free. The first-step diagnostic metrics were successfully
logged, including `train/train_rollout_kl=0.00234149`; the run is not a valid
two-step smoke because the post-train update failed.

### FP8 KV-cache scale warning affects absolute evaluation quality

SGLang reports that the checkpoint does not provide FP8 KV-cache scaling
factors and uses scale 1.0. Both comparison arms use the same rollout
checkpoint and setting, so the warning does not change the controlled
train-precision comparison. It can affect absolute AIME accuracy and must be
stated with the final curves.

### GB300 rollout kernels use an untuned fallback configuration

During SGLang CUDA-graph capture, both accuracy arms report that the matching
GB300 FP8 W8A8 Triton MoE configuration files are absent and that the default
kernel configuration is used. This is not an accuracy-comparison difference:
both arms use the same FP8 checkpoint, rollout topology, backend selection,
and fallback. It is a rollout-performance limitation to revisit separately,
not a reason to change this controlled accuracy run.

### Megatron checkpoint writer SIGSEGV is independent of MSC

BF16 job `421560` completed all four online weight updates and all four train
steps. The fourth update completed in 151.8 seconds, all eight rollout engines
resumed, and the final train step completed in 153.1 seconds. This validates
the corrected producer/consumer CUDA IPC lifetime across the required
multi-update boundary, but the job is not a successful checkpointed segment.

At `2026-08-11 02:08:48 UTC`, rank 0 entered `save_model` for iteration 3.
One of its checkpoint writer children exited with status 0 and the other
exited with signal 11 immediately after printing:

```text
!!!!!!! Segfault encountered !!!!!!!
  File "/root/go/pkg/mod/golang.org/toolchain@v0.0.1-go1.25.10.linux-arm64/src/runtime/sys_linux_arm64.s", line 440, in runtime.sigfwd
```

The parent remains blocked in `MegatronTrainRayActor.save_model` because the
failed writer never completed the multiprocessing queue acknowledgement. The
partial `iter_0000003` directory contains 64 `.distcp` shards totaling about
3.7 TiB, but it has neither `.metadata` nor
`latest_checkpointed_iteration.txt`; it is invalid and cannot be resumed.

The run used `ckpt_format=torch_dist`, `async_save=False`, fully parallel save,
and Megatron Core's MultiStorageClient path (`enable_msc=True`). That path
opens files through `multistorageclient` in forked checkpoint writer processes;
the signal-11 child and Go `runtime.sigfwd` stack identify this as the save
failure. The retry disables that path with Megatron's existing `--disable-msc`
flag and uses ordinary POSIX writes to Lustre. The hung allocation must be
canceled and its incomplete checkpoint removed before retrying. Rank 0's actor
stdout and stderr were first preserved under the run's `diagnostics` directory;
job `421560` was then canceled and the invalid 3.7 TiB checkpoint was removed.
Replacement BF16 job `422104` uses the same controlled configuration with
`--disable-msc`, run ID
`dsv4-aga-grpo-dapo-aime-bf16t-fp8r-60step-ipcgc-nomsc-s04-20260811T092227Z`,
and W&B ID `agabf16nomsc0811092227`. Replacement FP8 job `422108` uses run ID
`dsv4-aga-grpo-dapo-aime-fp8t-fp8r-60step-ipcgc256-nomsc-s04-20260811T092423Z`
and W&B ID `agafp8nomsc0811092423`. Both replacements received eight-node
allocations immediately. The same flag must be present in both comparison
arms; FP8 job `421770` was launched before this failure was discovered and
therefore cannot provide a valid checkpoint even if all four train steps
finish. It nevertheless completed all four FP8 train steps and four online
updates, then reproduced the identical failure: one checkpoint writer exited
0, one exited 11 in Go `runtime.sigfwd`, and the parent deadlocked in
`save_model`. Its 64 partial shards also totaled about 3.7 TiB and lacked both
metadata and the iteration tracker. Rank 0's logs were archived under the run's
`diagnostics` directory before canceling job `421770` and deleting the invalid
checkpoint.

The four diagnostic KL values from `421770` were `0.00354038`, `0.00344316`,
`0.00337105`, and `0.00332863`; the corresponding absolute train/rollout
log-probability differences were `0.0284232`, `0.0271491`, `0.0265690`, and
`0.0261787`. Its rollout-batch pass@1 values were `0.558594`, `0.453125`,
`0.496094`, and `0.507812`. These metrics remain valid four-step diagnostics,
but the run cannot be continued because it has no valid checkpoint.

BF16 replacement `422104` exposed an independent launch-cleanup defect before
the initial weight update. Seven SGLang servers became healthy, but the server
on `10.49.68.154` failed to bind TCP port 15000 with `Errno 98: address already
in use`. Its Ray `SGLangEngine` actor remained `ALIVE` after Uvicorn shut down,
while the rollout manager stayed `PENDING_CREATION`; aggregate stdout therefore
looked like slow initialization rather than a failed engine. The per-actor log
was archived under the run's `diagnostics` directory and job `422104` was
canceled without producing training data or a checkpoint.

The wrapper previously ran `ray stop --force` on every node, while the recipe's
`pkill sglang` command ran only on the head. Interrupted SGLang multiprocessing
children can outlive Ray and retain the fixed ports used by the next exclusive
allocation. The wrapper now kills orphan processes whose command names begin
with `sglang::` and kills any remaining listeners on ports 15000--15003 on
every node before starting Ray. BF16 replacement `422193` uses this cleanup,
`--disable-msc`, run ID
`dsv4-aga-grpo-dapo-aime-bf16t-fp8r-60step-ipcgc-nomsc-portclean-s04-20260811T094626Z`,
and W&B ID `agabf16nomscp0811094626`.

Jobs `422108` and `422193` then exposed that the SGLang-owned clone plus device
synchronization was not a deterministic lifetime boundary. FP8 job `422108`
completed train steps 0--2. Its post-update rank-0 free memory was 191.75,
189.73, then 137.65 GiB; the third update left about 52 GiB of additional
producer allocation live. The fourth update emitted PyTorch's warning that the
producer was trying to deallocate more than 1,000 blocks still referenced by a
consumer, then OOMed on a 130 MiB flattened bucket with 97 MiB free. BF16 job
`422193` completed step 0, then showed the same warning during its second update
and OOMed on a 34 MiB bucket with 23 MiB free. Both jobs exited 1 at about
03:33 UTC and produced no checkpoint.

The clone prevents the model loader from retaining the producer's IPC storage,
but deleting local tensor names does not force Python to collect all request and
loader reference cycles. Whether ordinary generational GC ran between updates
explains why jobs `421560` and `421770` crossed four updates while these retries
did not. The experiment patch now performs one device synchronization,
`gc.collect()`, `torch.cuda.ipc_collect()`, and `torch.cuda.empty_cache()` on
every SGLang TP worker in `end_weight_update`. This is once per complete model
update, not once per bucket. The launcher syntax-checks the patched source and
asserts that both cleanup calls are present before Ray starts.

The first GC-enabled submissions, `422364` and `422365`, failed during container
startup rather than training. The updated patch had been copied to
`experiments/aga_dsv4_flash/sglang_tensor_update_cuda_sync.patch`, while the
launcher correctly read the older file under the `patches/` subdirectory. Its
new assertion rejected that stale source. Importing the full SGLang module only
to verify a source edit also initialized optional modules and produced unrelated
import failures on some nodes. The launcher now syntax-checks the patched file
and verifies the two cleanup calls directly, without importing SGLang. A
checksum-forced sync put the new patch at the mounted path; one-node container
job `422387` then applied it and printed the expected cleanup block before
exiting 0. Jobs `422391` and `422392` are the clean paired retry.

FP8 job `422391` was canceled by cluster service account
`svc-hwinf-cs-sched` after 30 minutes while all 32 SGLang schedulers were doing
CPU-side Inductor/Triton compilation and the GPUs were temporarily idle. It had
no traceback and had not reached the initial online update. `scontrol` showed
that the intended idle-GPU exemption comment had lost its JSON quotes during
`#SBATCH` parsing, so the reaper did not recognize the requested 180-minute
model-loading exemption. The directive now wraps the complete JSON value in
single quotes, and FP8 replacement `422581` also supplies the valid JSON through
the `sbatch --comment` argument. `scontrol` confirms that the stored comment
retains all JSON quotes. BF16 job `422392` continues independently.

BF16 job `422392` subsequently completed all four online updates, rollouts,
and train steps without a CUDA IPC warning. Rollout pass@1 was `0.562500`,
`0.523438`, `0.546875`, and `0.484375`; diagnostic train/rollout KL was
`0.00241481`, `0.00237874`, `0.00234019`, and `0.00218904`. The cold first
train step took 1,226.75 seconds and the three steady-state steps took 154.21,
153.63, and 153.52 seconds.

Its final POSIX save then proved that disabling MSC does not fix the checkpoint
failure. The run had `async_save=False` and `--disable-msc`, but two child
processes under a representative `MegatronTrainRayActor.save_model` became
zombies with exit code 11. All 32 trainer actors were sleeping in
`futex_wait_queue` inside `save_model`; the 3.7 TiB `iter_0000003` payload had
neither `.metadata` nor `latest_checkpointed_iteration.txt`. Job `422392` was
canceled after 11 minutes without save progress and the invalid payload was
deleted. The failure is in the remaining Megatron/PyTorch distributed
checkpoint subprocess path, not specifically in MultiStorageClient.

FP8 replacement `422581` completed all four online updates, rollouts, and train
steps without a CUDA IPC warning or CUDA OOM. Rollout pass@1 was `0.589844`,
`0.484375`, `0.546875`, and `0.515625`; diagnostic train/rollout KL was
`0.00334554`, `0.00329539`, `0.00329482`, and `0.00321799`. The cold first
train step took 1,414.84 seconds and the three steady-state steps took 238.42,
239.95, and 242.62 seconds. Updates 2--4 each completed in about 65 seconds.

Its final POSIX save reproduced the same failure as `422392`. Two checkpoint
children under a representative trainer actor were zombies with exit code 11,
while the parent remained blocked in `save_model`. The actor stdout/stderr were
preserved under
`diagnostics/checkpoint_save_hang_job422581`; job `422581` was then canceled
and its invalid 3.7 TiB payload was deleted. Both four-step training segments
are complete in W&B, but neither has a resumable checkpoint.

## Active jobs

The experiment scope was capped at the current four-step paired validation on
2026-08-11. No 12/20/.../60-step continuation or additional checkpoint-only
retry will be submitted. Checkpoint validity is reported separately from the
four completed training steps; BF16 completed the training validation but its
final save failed. Some existing run IDs retain `60step` from the original
long-run plan; that string is no longer the intended stopping target.

| Job | Purpose | State at last update |
|---:|---|---|
| 414483 | FP8 train + FP8 rollout, two-step smoke with NCCL collective trace | Completed (0:0) |
| 414842 | FP8 train + FP8 rollout, phase-rendezvous two-step smoke | Failed: flattened weight bucket OOM at 1,660/2,130 |
| 414843 | BF16 train + FP8 rollout, phase-rendezvous two-step smoke | Failed: flattened weight bucket OOM at 1,823/2,130 |
| 415036 | FP8 train + FP8 rollout, 0.6 SGLang fraction retry | Failed after a successful 73.6-second update: one SGLang engine OOM during KV/graph resume |
| 415038 | BF16 train + FP8 rollout, 0.6 SGLang fraction retry | Failed at 1,025/2,130 in the post-train update due to CUDA IPC/bucket memory accumulation |
| 415487 | FP8 train + FP8 rollout, explicit post-update CUDA IPC cleanup | Completed (0:0), two steps; W&B `ap02gxq6` |
| 415660 | BF16 train + FP8 rollout, explicit post-update CUDA IPC cleanup | Completed (0:0), two steps; W&B `calfx8oo` |
| 416113 | FP8 train + FP8 rollout, original continuous 60-step submission | Canceled before start after about seven hours pending; replaced by checkpointed short-QoS segments |
| 416191 | BF16 train + FP8 rollout, original continuous 60-step submission | Canceled before start after about seven hours pending; replaced by checkpointed short-QoS segments |
| 419778 | FP8 train + FP8 rollout, first segmented accuracy attempt | Failed (1:0) during the fourth update around bucket 864/2,130; no checkpoint; W&B `agafp80810195339` |
| 419777 | BF16 train + FP8 rollout, first segmented accuracy attempt | Failed (1:0) during the third update at bucket 746/2,130; no checkpoint; W&B `agabf160810200400` |
| 420258 | FP8 consumer-sync validation launch | Failed before model launch: invalid wrapper `MODE=full`; no accuracy data |
| 420259 | BF16 consumer-sync validation launch | Failed before model launch: invalid wrapper `MODE=full`; no accuracy data |
| 420319 | FP8 train + FP8 rollout, four-step consumer-sync validation | Failed during the third update near bucket 1,120/2,130 with only 37 MiB free; consumer IPC warning persisted; W&B `agafp8ipc0811043818` |
| 420320 | BF16 train + FP8 rollout, four-step consumer-sync validation | Failed after step 0: all 2,130 update buckets completed, then one engine OOMed restoring its KV/graph pool because GPU optimizer offload was explicitly disabled; W&B `agabf16ipc0811043818` |
| 420703 | FP8 train + FP8 rollout, recipe-default optimizer-offload four-step run | Failed during the initial update at bucket 1,791/2,130; consumer IPC warning persisted; W&B `agafp8opt0811052946` |
| 420704 | BF16 train + FP8 rollout, recipe-default optimizer-offload four-step run | Failed during the initial update at bucket 1,731/2,130; consumer IPC warning persisted; W&B `agabf16opt0811052946` |
| 420965 | FP8 train + FP8 rollout, SGLang-owned bucket four-step validation | Canceled before model launch after the paired submission omitted `ALL` from `sbatch --export` |
| 420966 | BF16 train + FP8 rollout, SGLang-owned bucket four-step validation | Failed before model launch: wrapper-computed `HEAD_NODE` was absent from the nested `srun` environment |
| 420997 | FP8 train + FP8 rollout, SGLang-owned bucket plus CPU optimizer | Failed near the end of step 0 when Ray host memory reached 880.22/920.00 GiB; initial update passed; KL `0.00351571`; no checkpoint; W&B `agafp8ipc30811062230` |
| 420998 | BF16 train + FP8 rollout, SGLang-owned bucket plus CPU optimizer | Failed near the end of step 0 when Ray host memory reached 878.44/920.00 GiB; initial update passed; KL `0.00236877`; no checkpoint; W&B `agabf16ipc30811062230` |
| 421318 | FP8 train + FP8 rollout, SGLang-owned bucket + GPU optimizer + NVMe trainer offload | Failed after the 73.8-second initial update: the visible head-node trainer rank retained about 164 GiB and its TP4 SGLang engine OOMed while restoring KV/graph memory; Ray dedup hid the other per-rank values; no checkpoint; W&B `agafp8gopt0811071530` |
| 421319 | BF16 train + FP8 rollout, SGLang-owned bucket + GPU optimizer + parameter rematerialization | Failed during the initial update around bucket 1,644/2,130 with only 109--129 MiB free; no IPC warning and no checkpoint; W&B `agabf16gopt0811071530` |
| 421559 | FP8 train + FP8 rollout, corrected producer IPC cleanup + GPU optimizer + NVMe trainer offload | Canceled after a collective stall at 8,009/8,537 initial 64 MiB buckets; no OOM, eval, or checkpoint; W&B `agafp8ipcgc0811075257` |
| 421560 | BF16 train + FP8 rollout, corrected producer IPC cleanup + GPU optimizer + parameter rematerialization | All four updates and train steps completed, then checkpoint save deadlocked after an MSC writer SIGSEGV; canceled after preserving diagnostics and removed the invalid 3.7 TiB checkpoint; W&B `agabf16ipcgc0811075257` |
| 421770 | FP8 train + FP8 rollout, corrected producer IPC cleanup + GPU optimizer + NVMe trainer offload | All four updates and train steps completed, then reproduced the MSC writer SIGSEGV and save deadlock; diagnostics preserved, invalid 3.7 TiB checkpoint deleted, job canceled; W&B `agafp8ipcgc2560811081817` |
| 422104 | BF16 train + FP8 rollout, MSC-disabled checkpoint retry | Canceled before the initial update after one SGLang actor failed to bind stale port 15000; diagnostics preserved; no training data or checkpoint; W&B `agabf16nomsc0811092227` |
| 422108 | FP8 train + FP8 rollout, MSC-disabled checkpoint retry | Failed (1:0) during update 4 after step 2; consumer IPC references accumulated, then a 130 MiB bucket OOMed with 97 MiB free; no checkpoint; W&B `agafp8nomsc0811092423` |
| 422193 | BF16 train + FP8 rollout, MSC-disabled checkpoint retry with per-node port cleanup | Failed (1:0) during update 2 after step 0; same consumer IPC-reference warning, then a 34 MiB bucket OOMed with 23 MiB free; no checkpoint; W&B `agabf16nomscp0811094626` |
| 422364 | FP8 train + FP8 rollout, forced consumer GC and POSIX checkpoint validation | Failed before Ray startup: launcher assertion detected that the mounted patch was still the older version; no W&B training data |
| 422365 | BF16 train + FP8 rollout, forced consumer GC and POSIX checkpoint validation | Failed before Ray startup for the same stale mounted patch; no W&B training data |
| 422376 | One-node SGLang patch inspection | Failed as intended: confirmed the launcher-visible patch did not contain the GC hunk |
| 422387 | One-node SGLang patch inspection after checksum sync | Completed (0:0): patched container source contains the complete end-of-update cleanup |
| 422391 | FP8 train + FP8 rollout, verified consumer-GC patch and POSIX checkpoint | Canceled by the cluster idle-GPU reaper during CPU-side CUDA-graph kernel compilation because the exemption comment was malformed; no training data; W&B `agafp8gc20811105515` |
| 422392 | BF16 train + FP8 rollout, verified consumer-GC patch and POSIX checkpoint | Four update/rollout/train steps completed without IPC warning; POSIX save still reproduced writer SIGSEGV/deadlock, so job was canceled and invalid 3.7 TiB payload deleted; W&B `agabf16gc20811105516` |
| 422581 | FP8 train + FP8 rollout with verified reaper exemption | Four update/rollout/train steps completed without IPC warning or OOM; POSIX save reproduced writer SIGSEGV/deadlock, so job was canceled, actor logs archived, and invalid 3.7 TiB payload deleted; W&B `agafp8gc30811112908` |

The two original full jobs were initially submitted with a 36-hour limit, then
reduced while pending to 12 hours based on smoke timings. Neither job started.

### Full-run queue delay is scheduler pressure, not a recipe failure

Jobs `416113` and `416191` remained eligible but pending with reason `Priority`
for about seven hours. Across inspections, Slurm had roughly 925--998 pending
jobs; 365--406 had higher priority than these runs, while backfill cycles timed
out after considering only 76--138 candidates. Although
`batch_long` reported idle nodes, reservations, priority, `SegmentSize=8`, and
the single-switch request constrained usable eight-node placements. Both jobs
had valid account/QoS/TRES requests and were reevaluated by the scheduler.

The continuous submissions were canceled before start and replaced with
four-step checkpointed segments under the account's standard `short` QoS. That
QoS has a two-hour wall limit but enough priority to enter the observed
backfill depth. Each completed segment saves model, optimizer, RNG, and global
dataset cursor state, then the next segment resumes the same run ID and W&B ID
with a larger `--num-rollout` target until 60.

This recipe has `partial_rollout=False`. Aborted oversampling requests are
discarded rather than retained in the data-source buffer, so no dynamic-sample
buffer is lost at a segment boundary. A resumed segment does run one additional
read-only AIME evaluation before its first training rollout; this adds curve
points and runtime but does not change model or optimizer state. Only the next
pair is submitted after the current pair passes checkpoint/resume validation,
so the queue is not filled with dependent jobs.

The first short-QoS pair started together at approximately 2026-08-11 03:18
UTC, about 12 minutes after submission. Both allocations have eight GB300
nodes and reached an eight-node Ray cluster. The expanded launch commands
confirm the intended controlled comparison: GRPO on DAPO Math 17k, AIME-2024
evaluation with eight samples per problem, TP2/PP8/CP2/EP4 training, eight
TP4/EP4 rollout engines, and a four-rollout stopping target for this segment.

Both initial online updates completed across all 32 trainer ranks and all eight
rollout engines: 74.3 seconds for BF16 training and 71.4 seconds for FP8
training. The initial AIME-2024 evaluation produced pass@1/pass@8 of
`0.4667`/`0.5667` for the BF16-training arm and `0.4500`/`0.5667` for the
FP8-training arm. The pass@1 difference is one half-success per problem on
average over only 30 problems and is not a conclusion; later paired eval
points are required.

Ray's aggregated Slurm stdout stopped flushing one arm's progress partway
through evaluation even though the actor continued normally. The authoritative
per-actor log under the Ray session showed that evaluation and rollout 0 had
completed, and the trainer processes and GPU state confirmed train step 0 was
active. When aggregate stdout appears stale, inspect the matching Ray actor
log before classifying the run as stalled.

Job `420703` briefly received an earlier allocation and wrote only its wrapper
header before Slurm returned it to `PENDING`; no `srun`, model process, or W&B
run started in that allocation. It later started normally together with
`420704`. In both active allocations the patch applied on all eight nodes, and
the expanded trainer command contains `--optimizer-cpu-offload`,
`--use-precision-aware-optimizer`, and
`--overlap-cpu-optimizer-d2h-h2d`, confirming that the official recipe setting
was active rather than merely recorded in the wrapper environment. Both jobs
subsequently failed during the initial online update as described above; no
training step or checkpoint was produced.
