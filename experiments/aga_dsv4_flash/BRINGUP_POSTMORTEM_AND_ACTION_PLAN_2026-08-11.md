# DeepSeek-V4-Flash RL Bring-up: Postmortem and Action Plan

Date: 2026-08-11

## Executive summary

The four-step online RL computation now works for both controlled arms:

- BF16 training + FP8 rollout
- blockwise FP8 training + FP8 rollout

Both arms completed four online weight updates, four rollouts, and four train
steps without a CUDA IPC warning or CUDA OOM. The remaining correctness blocker
is checkpoint saving: both final saves produced invalid 3.7 TiB payloads and
deadlocked after checkpoint child processes exited with signal 11.

The bring-up took too many attempts because launcher errors, resource-policy
errors, memory-capacity errors, CUDA IPC lifetime errors, and checkpoint errors
were discovered one at a time. Several only appear after the third update or
the final save, so a two-step smoke was not a sufficient gate.

The current path is usable for short, non-resumable diagnostics. It is not ready
for a long accuracy run or an upstream recipe claim until checkpoint save/load
works.

## Root-cause summary

| Area | Problem | Root cause | Current status | Permanent action |
|---|---|---|---|---|
| Scheduling | Original 60-step jobs waited about seven hours | Eight-node placement under long QoS had low priority and poor backfill opportunity | Avoided | Use short, checkpointed validation segments; submit a long run only after every gate passes |
| Idle-GPU policy | FP8 job was canceled during CPU-side kernel compilation | The `#SBATCH --comment` JSON lost its quotes, so the reaper exemption was ignored | Fixed in launcher | Quote the full JSON value and verify the stored comment with `scontrol` before model launch |
| Slurm environment | Nested `srun` lost `HEAD_NODE` | Explicit `--export` omitted `ALL`; wrapper-created variables were not propagated | Fixed in launcher | Always use `--export=ALL,...` and set `SLURM_EXPORT_ENV=ALL`; add a preflight assertion |
| Recipe mode | Two jobs exited before model launch | Submission used `MODE=full`; the supported full recipe mode is `normal` | Fixed manually | Replace free-form mode strings with validated CLI choices and a dry-run command |
| Runtime cleanup | One SGLang server could not bind port 15000 | A prior interrupted allocation left SGLang children/listeners alive | Fixed in launcher | Kill stale Ray/SGLang processes and fixed-port listeners on every node before startup |
| Patch deployment | Correct SGLang patch was copied to the wrong directory | Launcher read `patches/`, while the updated file was synced to its parent directory | Fixed manually | Stop runtime patching for production; build a pinned image. Until then, verify exact path and SHA-256 on every node |
| Image reproducibility | Runtime was `radixark/miles:latest` | The image tag was recorded without a digest | Open | Pin and report the image digest together with Miles, SGLang, Megatron, TE, CUDA, and NCCL revisions |
| Collective ordering | Colocated weight update intermittently stopped across PP/EP/TP phases | Ranks entered communicators with different membership in different host-side epochs | Fixed in experiment | Keep the trainer-wide Gloo rendezvous and add a repeated mixed-topology integration test |
| Update bucket size | 1 GiB buckets stalled; 64 MiB was excessive for FP8; 256 MiB lacked BF16 headroom | One bucket size was not suitable for both precision/memory layouts | Tuned | Use 64 MiB for BF16 and 256 MiB for FP8 on this topology; expose the values as recipe fields |
| GPU headroom | Weight transfer or SGLang resume OOMed | SGLang static memory fraction 0.7 left too little colocated transfer headroom | Tuned | Use 0.6 for this 4-GPU-per-node colocated topology and validate free-memory margin before update |
| Host memory | Ray killed trainers near 880/920 GiB | CPU optimizer offload plus four trainers and SGLang exceeded node RAM | Resolved by configuration | Keep optimizer state on GPU. Use BF16 parameter rematerialization and FP8 NVMe trainer offload |
| CUDA IPC lifetime | Third/fourth updates accumulated over 1,000 producer-owned CUDA IPC blocks and OOMed | SGLang retained deserialized producer storage after the HTTP response; device sync and `del` alone did not sever ownership | Fixed in experiment, not upstream | Clone the bucket into SGLang-owned CUDA storage, release the IPC tensor, and run end-of-update IPC/allocator cleanup |
| Validation depth | Two-step smoke passed, formal run later failed | Two steps execute only two weight updates and never test the known third/fourth-update boundary | Process failure | Make at least four update cycles mandatory before any accuracy run |
| Checkpoint | Both final saves deadlocked with invalid 3.7 TiB payloads | Checkpoint children exited with signal 11 even with `async_save=False` and `--disable-msc`; the parent waited forever | Open blocker | Build a save-only reproducer, capture a native core, identify the remaining subprocess/Go-linked path, and provide an inline synchronous writer |
| Checkpoint state | Rank 28 warned that its common state dict differed from rank 0 immediately before the crash | Rank-local `args` content is not identical across checkpoint ranks | Open | Diff and canonicalize the serialized common state before debugging the native crash; fail before writing if common state differs |
| W&B teardown | A canceled run can remain displayed as `running` temporarily | SIGTERM during the save deadlock bypasses normal `wandb.finish()` | Minor | Add a termination handler that flushes W&B and records the run as failed after preserving the last metrics |
| Absolute eval quality | SGLang used FP8 KV-cache scale 1.0 | The checkpoint does not contain calibrated KV-cache scale factors | Open accuracy limitation | Calibrate/provide KV-cache scales or use BF16 KV cache for an absolute-accuracy run |
| GB300 rollout performance | SGLang selected default FP8 W8A8 MoE configs | Matching tuned GB300 kernel configs were absent | Open performance limitation | Add tuned GB300 configs after correctness work; this did not bias the controlled precision comparison |

## Code changes required

### Miles

1. Upstream the producer-side lifecycle change in
   `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`:
   keep the CUDA IPC bucket alive until every engine acknowledges import, use a
   global rendezvous before the next collective epoch, release bucket
   references, then run `synchronize -> gc.collect -> ipc_collect ->
   empty_cache` once per complete update.
2. Add an integration test that performs at least five consecutive online
   updates with the DeepSeek PP8/EP4/TP2 communicator layout. Check that all
   engines generate after every update and producer free memory stops drifting
   after warm-up.
3. Add a clear non-resumable smoke mode that skips the final checkpoint. A
   four-step online-RL test should not spend 3.7 TiB and hang on a checkpoint
   path already known to be broken.
4. Make the launcher schema typed and fail-fast. Validate mode, precision,
   bucket size, memory fraction, W&B configuration, patch hashes, required
   paths, exported variables, and the Slurm reaper comment before Ray starts.
5. Record one run manifest containing git revisions, dirty-patch hashes, image
   digest, dataset/model hashes, generated command line, topology, and W&B ID.

### SGLang

1. Upstream the consumer-side ownership boundary in
   `SchedulerWeightUpdaterManager.update_weights_from_tensor`: clone a received
   `flattened_bucket` into SGLang-owned CUDA storage before model loading,
   synchronize before releasing the IPC-backed source, and perform final
   garbage/IPC/allocator cleanup in `end_weight_update`.
2. Add a CUDA IPC regression test with an external producer. Run at least five
   update cycles and assert that the producer receives each acknowledgement
   only after its source storage is safe to release.
3. Longer term, replace implicit Python-object lifetime with an explicit
   ownership/lease protocol. A zero-copy path is acceptable only if SGLang
   returns a completion event or acknowledgement that proves every consumer
   stream has retired the source allocation.

### Megatron Core / PyTorch distributed checkpoint

1. Create a save-only 32-GPU reproducer: load the same distributed checkpoint,
   build model and optimizer state, call one save, and exit. Do not launch
   SGLang, rollout, or training.
2. Run a small isolation matrix:
   - model state only versus model + optimizer + RNG state;
   - fully parallel save enabled versus disabled;
   - MSC enabled versus disabled;
   - checkpoint subprocess writer versus an inline synchronous writer.
3. Resolve the common-state mismatch first. Print a structured diff of the
   rank-local `args` objects, remove rank-local/runtime-only fields from common
   state, and require all ranks to agree before any shard is written.
4. Capture the native failure with `ulimit -c unlimited`, Python fault handling,
   and `GOTRACEBACK=crash`. Preserve the child process maps and obtain a
   symbolized native backtrace. The observed stack enters Go
   `runtime.sigfwd`, so confirm which Go-linked library remains loaded when MSC
   is disabled instead of assuming that `--disable-msc` removed it.
5. Provide a true inline synchronous writer as a correctness fallback. The
   current `async_save=False` setting still creates multiprocessing children,
   so it does not provide that guarantee.
6. Define checkpoint success as all of the following: writer exits normally,
   `.metadata` exists, `latest_checkpointed_iteration.txt` points to iteration
   3, a new job loads it, optimizer/RNG state restores, and selected parameter
   hashes match.

### Cluster and launcher operations

1. Build and pin an experiment image instead of applying source patches on each
   node at startup.
2. Split preflight from GPU execution. Command generation, path/hash checks,
   W&B authentication, Slurm JSON validation, and image/source verification do
   not need 32 GPUs.
3. Keep one short validation ladder:
   - one-node container/bootstrap test;
   - 32-GPU save-only checkpoint test;
   - 32-GPU five-update online-transfer test;
   - paired four-step RL validation;
   - only then, a longer accuracy run.
4. Stop a failed phase immediately. Do not keep an allocation alive for a
   known distributed timeout when logs, process states, and GPU topology already
   prove the failure mode.

## Acceptance gates for the next run

Do not start another accuracy run until all P0 gates pass:

| Priority | Gate | Pass condition |
|---|---|---|
| P0 | Checkpoint save/load | Save exits 0, metadata/tracker exist, and a fresh job reloads iteration 3 with optimizer state |
| P0 | Online update lifetime | Five consecutive updates, no IPC warning/OOM, all engines resume, and post-warm-up producer free-memory drift is below 2 GiB |
| P0 | Reproducibility | Pinned image digest and complete source/config/data manifest |
| P0 | Launcher preflight | Invalid mode/export/comment/path/patch conditions fail before model initialization |
| P1 | Absolute evaluation | Calibrated FP8 KV-cache scales or BF16 KV cache for the accuracy baseline |
| P2 | Performance | Profile BF16 and FP8 only after P0/P1 correctness gates pass |

## Recommendation

Contribute the Miles and SGLang online-update ownership fixes now; the paired
four-step result validates them across the previously failing lifecycle. Do not
present the complete DeepSeek-V4-Flash RL recipe as resumable or long-run-ready
until the checkpoint SIGSEGV is fixed and a save/load test passes.

For short diagnostics, disable final save and use the validated four-step path.
For an accuracy claim, fix checkpointing and KV-cache calibration first, then
run a longer paired curve. The experiment reported here used blockwise FP8, not
MXFP8.

## Evidence

- Four-step result: `experiments/aga_dsv4_flash/FOUR_STEP_RESULTS_2026-08-11.md`
- Detailed chronological log: `experiments/aga_dsv4_flash/EXPERIMENT_LOG_2026-08-10.md`
- BF16 W&B: https://wandb.ai/megatron-core-moe-dev/miles-run_deepseek_v4/runs/agabf16gc20811105516
- FP8 W&B: https://wandb.ai/megatron-core-moe-dev/miles-run_deepseek_v4/runs/agafp8gc30811112908
- Final jobs: BF16 `422392`, FP8 `422581`

