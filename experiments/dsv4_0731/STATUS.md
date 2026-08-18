# DeepSeek-V4-Flash-0731 MXFP8 RL bring-up

Last updated: 2026-08-18

## Objective

Two phases, both training with the Megatron MXFP8 recipe on 32 GB300 GPUs:

1. **MXFP8 train + MXFP8 rollout.** Rollout serves an MXFP8 checkpoint derived
   from the release, so both sides use the same nominal format.
2. **MXFP8 train + native rollout.** Rollout serves `DeepSeek-V4-Flash-0731`
   as shipped — MXFP4 routed experts against MXFP8 activations. The
   train/rollout mismatch is expected to rise relative to phase 1.

Both phases share one trainer checkpoint, so the mismatch delta isolates the
rollout-side weight format.

## 2026-08-18 direct-HF no-R3 OCI smoke passes

P1 and P2 can now complete a one-step training smoke without an offline BF16 HF
conversion or a `torch_dist` seed. Both jobs used the official hybrid HF
checkpoint as the trainer source (`init_model_source=hf`), trainer-owned online
rollout weights (`rollout_weight_source=trainer`) and SGLang dummy loading. P1
used a 6.2 MB metadata-only MXFP8 schema with no tensor/index payload; P2 used
the official config and packed-MXFP4 expert layout directly.

| Phase | OCI job | R3 | Initial update | Log-prob | Train | Post-train update | Final state |
|---|---:|---|---:|---:|---:|---:|---|
| P1: MXFP8 rollout | `501870` | off | 428--430 s | 688 s | 1520--1522 s | 225--226 s | `COMPLETED 0:0`, 59:25 |
| P2: FP8 + packed-MXFP4 experts | `501657` | off | 388--389 s | 667 s | 1492--1495 s | 163--164 s | `COMPLETED 0:0`, 54:57 |

Each job ran on 8 OCI nodes / 32 GPUs, generated a real 256-sample rollout,
completed `compute_log_prob`, performed one optimizer step and synchronized the
updated weights a second time. All 32 ranks reported both update transactions
successful. The P2 job applied only the SGLang TRT-LLM MXFP4 hot-reload and
memory-saver patches; the separate R3 `HashTopK` patch was not applied.

The rollout metrics rule out the earlier corruption:

| Metric | P1 `501870` | P2 `501657` |
|---|---:|---:|
| `raw_reward` | 0.546875 | 0.710938 |
| `truncated_ratio` | 0.558594 | 0.371094 |
| `repetition_frac` | 0.0 | 0.0 |
| `train_rollout_kl` | 0.126726 | 0.131041 |

The first three values show that P2 no longer has zero reward, near-100%
truncation or repetitive garbage. The KL values are still much higher than the
older four-step reference runs, so this smoke does not establish numerical
parity.

The scope is deliberately limited: one optimizer step, no generation after the
second reload, no save/resume and no R3. Both jobs also printed CUDA IPC unlink,
NCCL process-group and W&B broken-pipe atexit stacks after Ray had declared
success. Slurm exited `0:0` in both cases, making these teardown defects rather
than runtime failures, but they remain cleanup work.

Validated revisions are Miles `5cb48105e`, Megatron-Bridge `ca1a03de`, and
SGLang MR !1 head `605830cd5c`. R3 is split into independent SGLang MR !2 at
`8007bb9d31` and was not part of this result.

## 2026-08-17 direct-HF initial sync passes on local B300

Fresh P1/P2 runs no longer require the 567 GB BF16 HF artifact or an offline
`torch_dist` seed. The new `init_model_source=hf` path imports the official
hybrid checkpoint through Megatron-Bridge at trainer startup, while
`rollout_weight_source=trainer` starts SGLang in dummy-load mode and obtains all
weights from the initial online sync.

P1 allocates from a metadata-only MXFP8 schema; P2 allocates the official
FP8/MXFP4 layout and uses `flashinfer_mxfp4`. On the current 8×B300 node, two
TP4/EP4 engines completed every bucket, both engines returned 200 from
`end_weight_update` and `continue_generation`, all eight trainer ranks reported
`update_weights phase=end ok=true`, and both containers exited zero. P1 took
about 285--287 s for the update. P2 took 615--617 s, including about 358 s of
one-time MHC/TileLang compilation.

This local run deliberately used zero optimizer steps. It validates direct HF
load, Bridge export/quantization, atomic weight/scale grouping, dummy rollout
allocation and MXFP4 finalization. The OCI smoke above adds one optimizer step
and a second reload, but not broader numerical parity, repeated post-update
generation or checkpoint resume.

## 2026-08-17 root cause confirmed: unregistered MXFP4 clamp tensor

The packed-MXFP4 update failure now reproduces and is fixed on the current local
B300 node. The differentiating flag missing from the earlier local controls was
`enable_memory_saver=True`.

`Mxfp4FlashinferTrtllmMoEMethod.create_moe_runner` allocates
`_gemm1_clamp_limit_tensor` as a plain CUDA tensor, with one `10.0` value per
local expert. Model construction occurs inside TorchMemorySaver's `weights`
region. With `enable_weights_cpu_backup=False`, pausing and resuming that region
preserves the virtual address but not the data. The tensor is not a parameter or
buffer, so `_export_static_state`, the Miles full update and WeightChecker all
miss it; `apply()` still passes it to every TRT-LLM MXFP4 MoE call as
`gemm1_clamp_limit`.

The causal chain is measured at four levels:

| Measurement | Result |
|---|---|
| standalone TorchMemorySaver tensor in the `weights` region | `[10.0]` before pause/resume, `[0.0]` after |
| real 64-expert frozen-input TRT-LLM MXFP4 op | clamp 10 digest `cbbad8e8e1fe2710`; clamp 0 digest `07854d2fef297a06`, relative L2 1.0; restoring 10 is bitwise exact |
| `dsv4-miles-initial-sync-memsaver-hchead`, graph enabled | 67,569 tensors; checker `Success`; first generation `4/4` truncated at 1024 |
| `dsv4-miles-initial-sync-memsaver-nograph` | same with `disable_cuda_graph=True`; checker `Success`; `4/4` truncated |
| `dsv4-miles-initial-sync-memsaver-clampfix` | register the tensor as a non-persistent buffer; checker `Success` including all 43 clamp buffers; token counts 26/45/21/124, all `finish=stop`, `0/4` truncated |

All three end-to-end containers exited zero. The failing generations repeat
prompt fragments just like the OCI jobs. The fixed container changes no weight,
quantization, transport, graph or kernel code; it only registers the existing
tensor on the expert layer with `persistent=False`. SGLang's existing
memory-saver static-state export/import then backs it up and restores it, and
the checker covers it.

This identifies the root cause. CUDA graph, visible weights, rank-local bucket
transport, MXFP4 quantization and restore/repack are not the failure mechanism.
The OCI P2 smoke above verifies that the packed-MXFP4 path produces usable
rollouts with the updated SGLang MR. Longer numerical and lifecycle validation
remains.

## 2026-08-17 local B300 full-update control

The release runs locally on one B300 node with TP4/EP4,
`moe_runner_backend=flashinfer_mxfp4`, 1 GiB flattened CUDA-IPC buckets and one
begin/end transaction spanning all 48 shards. GPU 0-3 provide both the engine
ranks and their corresponding rank-local export buffers. The early controls
called the release/resume APIs but had `enable_memory_saver=False`, so those
calls were no-ops. The root-cause runs above correct that coverage gap.

An initial full replay produced `0/4` to `4/4` truncated generations, but its
weight checker exposed a harness error rather than the Phase 2 failure. The
release tensor `layers.N.attn.wo_a.weight` is FP8 `[8192, 4096]` with a 128x128
UE8M0 scale. With `SGLANG_OPT_FP8_WO_A_GEMM=0`, the runtime parameter is a BF16
`[2048, 4096]` TP shard. The harness copied FP8 numeric values directly into
BF16, yielding `max_abs_err=447.890625` and corrupting almost every element.
Real Miles does not quantize `wo_a` in this configuration; it sends the BF16
trainer weight.

Two controls isolated and then removed that error:

| Container / selector | Checker | Generation |
|---|---|---|
| `dsv4-experts-plus-nonexpert-weights-check` | fails; raw BF16 `wo_a` error up to 447.890625 | `0/4` to `4/4` truncated; invalid |
| `dsv4-experts-plus-nonexpert-weights-no-woa-check` | only small block-FP8 pair/layout differences | `0/4` to `0/4`; answers correct |
| `dsv4-full-miles-valid-woa-check`, `miles_full` | `Success` | `0/4` to `0/4`; all four texts identical |
| `dsv4-miles-initial-sync-full-strict`, `miles_full` before first generation | `Success` after 67,566 tensors, before KV/graph resume | first generation `0/4` truncated; all answers correct; exit 0 |
| `dsv4-miles-requantize-1p05-roundtrip-v3`, experts only | release expert -> BF16 x1.05 -> Miles MXFP4; 66,048 tensors | `0/4` after update; restore gives `0/4` and exact baseline text |
| `dsv4-miles-requantize-highbatch-strict-roundtrip`, experts only | four independent rank-local producers; batch-256 CUDA graph; strict restore checker `Success` | baseline/update/restore each `64/256` at the deliberate 64-token cap; all 192 completed answers restore; exit 0 |

`miles_full` reconstructs the initial BF16 master value of every `wo_a` by
multiplying each release FP8 block by its UE8M0 scale, omits the scale because
the runtime parameter is BF16, and converts the remaining scales to their Miles
update names/layouts. It sent 67,566 tensors. This is the same decode that
created the trainer's initial BF16 HF checkpoint, without loading Megatron or
taking an optimizer step.

The passing strict checker matters more than the behavioral smoke: it verifies
all visible expert and nonexpert parameters after the complete update. The
changed-value control additionally exercises the current workspace's real
`mxfp4_quantize`: every release expert was dequantized to BF16, multiplied by
1.05, and requantized. For one real expert, 44.8617% of weight bytes and
45.9473% of scale bytes changed, while the represented tensor changed by
relative L2 0.0206583. After all 66,048 expert tensors were updated, all four
answers remained correct and none reached the token limit. Restoring the
release recovered the four baseline generations byte for byte.

The rollout-sized run removes two remaining differences from that first
changed-value control. It uses four independently spawned CUDA producer
processes, one for each TP rank, and captures the decode graph through batch
size 256. Each process creates and retains its own flattened CUDA-IPC bucket
while the engine processes one begin/end transaction spanning all 48 shards.
The baseline, changed and restored generations each report `64/256` at the
64-token cap. Those 64 are copies of the intentionally open-ended fourth test
prompt; all other 192 requests finish, and their final answers match after
restore. Hidden-reasoning text is not fully deterministic at batch 256, so a
byte-for-byte text comparison is not a valid restore test there. The strict
weight checker is: it snapshots the finalized kernel layout before the update
and reports `Success` after restore.

The initial trainer sync happens before the first rollout and before any
optimizer step. A separate full-model control now follows that ordering: it
skips baseline generation, snapshots the fresh engine, runs the release ->
weights-only -> pause -> one full update -> continue sequence, strictly compares
the finalized model, restores KV cache and CUDA graph, and only then generates.
All 67,566 tensors load, the strict compare reports `Success`, all four first
generations answer correctly, and none reaches the 1024-token limit. This rules
out prior user generation as a prerequisite for a correct reload, but this
historical control had memory saver disabled and skipped the three `hc_head_*`
tensors. The 67,569-tensor root-cause runs above supersede it for the production
memory lifecycle.

The real Miles fused-FC1 conversion also has a checkpoint-byte control. A
release expert's `w1` and `w3` were decoded to BF16 and concatenated into the
Megatron `linear_fc1` layout. Running that tensor through `convert_to_hf` and
the MXFP4 processor reproduced both release weights and both UE8M0 scales
exactly. This verifies the gate/up split and quantizer entry point for a real
tensor, but not distributed PP/EP/TP gathering.

The same encoder check was extended over every packed expert in the release.
All 35,328 weights across the 46 expert-bearing shards re-encode byte for byte,
including all 33,024 main-model expert tensors the update sends and 2,304
speculative tensors it excludes. Their UE8M0 scales also match exactly. The
eight-GPU read-only check exits zero.

The kernel output itself now has a fixed-input control at the production
EP-rank size. All 64 real layer-0 local experts were shuffled into the TRT-LLM
layout and evaluated after fresh MXFP8 activation quantization and top-8 packing;
64 fixed tokens route across all 64 experts. Fresh repeat, identity hot reload,
and restore after a changed update were bitwise identical (`max_abs=0`, the
same output SHA-256 prefix `cbbad8e8e1fe2710`). Negating both E2M1 sign bits
changed the output with relative L2 1.5391512, so the op demonstrably consumed
the reloaded bytes. Zeroing only `_gemm1_clamp_limit_tensor` changed output by
relative L2 1.0 and restoring 10 was bitwise exact. The container
`dsv4-mxfp4-frozen-moe-output` exits zero.

The local evidence therefore clears the initial-value Miles-format payload,
ordinary post-master-weight value changes, the Miles MXFP4 quantizer, packed
MXFP4 restore/repack, fixed-input TRT-LLM MoE output, 1 GiB bucket transport,
independent producer processes, batch-256 graph/tactic selection and local
TP4/EP4 execution. The early controls did not clear the real memory cycle; the
root-cause section above shows that it is the trigger.
Inspection of the installed FlashInfer path further reduces the stale-state
hypothesis: autotuning caches tactic IDs but no tensor pointer; the raw op
constructs a launcher, runner and workspaces for each call and reads all current
tensor `data_ptr()` values during `prepare_moe`.

Distributed Megatron PP/EP/TP gather and eight-node colocated orchestration are
not required to trigger the failure: the production-order local replay now
reproduces it with the same image and actual memory saver. OCI job `501657`
subsequently verified the clamp-buffer fix in the full eight-node path; the next
useful probe is a multi-step run that generates after another post-train reload.

This is not explained by a newer local patch. The patched MXFP4 source in the
retained full-replay container is byte-identical to the current experiment
patch. Its last functional change is the address-stability fix that failing job
454574 already carried. The CUDA-IPC clone/synchronize patch predates the bad
jobs and is visibly active in the retained local container as well.

## Checkpoint layout of the release

`deepseek-ai/DeepSeek-V4-Flash-0731` is not uniformly quantized:

| Tensors | Storage | Scale |
|---|---|---|
| Routed experts (`w1`/`w2`/`w3`) | `I8`, two E2M1 elements per byte | `F8_E8M0`, one per 32 elements |
| Attention projections, shared expert | `F8_E4M3` | `F8_E8M0`, blockwise `[128, 128]` |
| Norms, embedding, router, `hc_*` | `BF16` / `F32` | none |

The preview checkpoint `sgl-project/DeepSeek-V4-Flash-FP8` differs on both
counts: its routed experts are unpacked `F8_E4M3`, and every scale is `F32`.

The 43-layer decoder is otherwise identical between the two. Only the
speculative module changes: the preview carries one EAGLE layer
(`enorm`/`hnorm`/`eh_proj`), the release carries three DSpark layers
(`main_proj`/`confidence_head`/`markov_head`). Megatron builds neither, since
`scripts/models/deepseek-v4-flash.sh` declares no MTP layers, so mbridge never
requests those tensors.

## Conversion

`tools/fp8_cast_bf16.py` now recognizes both layouts. It picks the branch from
the scale extent rather than the tensor name:

- `is_mxfp4_weight` — a scale covering `weight.shape[-1] * 2 // 32` groups means
  packed MXFP4; `mxfp4_dequant` unpacks it low nibble first and applies the
  UE8M0 block exponents.
- Everything else stays on the block-scaled FP8 path, with `to_float_scale`
  decoding UE8M0 scales that the preview stored as `F32`.

A missing `_scale_inv` for a quantized weight now raises instead of copying the
payload through, because an unconverted tensor is indistinguishable from a
successful cast downstream.

Both branches were checked against the SGLang runtime decode on a real shard and
match bit-for-bit — `cast_e2m1fn_to_e4m3fn` for MXFP4, `block_quant_dequant`
for FP8. See `validate_mxfp4_dequant.py`. `tests/fast/tools/test_fp8_cast_bf16.py`
covers the value table, block exponents, nibble order, and layout discrimination
without a GPU.

## Recipe wiring

`scripts/run_deepseek_v4.py`:

- `DeepSeek-V4-Flash-0731` registered under org `deepseek-ai`, reusing the
  `deepseek-v4-flash` Megatron model type and the 8-node SPMD conversion config.
- `rollout_fp4_experts()` decides whether the rollout checkpoint carries packed
  experts. It drives both `SGLANG_DSV4_FP4_EXPERTS` and the MoE runner backend.
  The env var used to be pinned to `0`, which would have forced an FP8 expert
  layout onto a packed checkpoint; setting it explicitly also suppresses
  SGLang's own header probe.
- Serving packed experts selects `flashinfer_mxfp4`, which resolves to the
  TRT-LLM kernel on SM100 and quantizes activations to MXFP8
  (`flashinfer_mxfp4_moe_precision` defaults to `default`).

`experiments/aws_dsv4_flash/run_rl.sbatch`:

- `PRECISION` accepts `mxfp8` (phase 1) and `mxfp8-fp4r` (phase 2).
- `MODEL_NAME` is now an input instead of a hardcoded preview name.

## Resulting artifacts

| Path under `models/` | Produced by | Consumed by |
|---|---|---|
| `DeepSeek-V4-Flash-0731` | `hf download` | phase 2 rollout |
| `DeepSeek-V4-Flash-0731-bf16` | `prepare-single` | the two conversions below |
| `DeepSeek-V4-Flash-0731-MXFP8` | `prepare-mxfp8` | phase 1 rollout |
| `DeepSeek-V4-Flash-0731_torch_dist` | `prepare-spmd` | trainer, both phases |

`prepare-spmd` needs a Ray cluster, so it runs inside the 8-node job via
`full-train`, which skips any stage whose output already exists.

## Run log

### 424410 — conversion, COMPLETED (9 m 59 s)

Produced the BF16 cast (567 G) and the MXFP8 rollout checkpoint (295 G).

The first attempt at this, job 424335, produced a BF16 cast that looked complete
but left `mtp.0.main_proj.{weight,scale}` in their source dtypes, and the MXFP8
stage then failed on the missing scale. The cast resolved scales through the
remapped tensor name, and the HF remap does not cover the DSpark speculative
module, so those two tensors fell through the branch that warned and copied.
Scale resolution now happens in the checkpoint's own namespace, a weight with no
resolvable scale raises, and `assert_fully_dequantized.py` re-reads the emitted
headers before the MXFP8 stage starts.

### 424444 — phase 1, FAILED (36 m 17 s)

Reached step 0 including the AIME eval, then one of the eight SGLang engines
(`nvl72d020-T04`) stopped making progress. All four of its TP ranks tripped the
300 s scheduler watchdog together, the server sent itself SIGQUIT, and the
trainer's next `release_memory_occupation` found the port closed:

```
TimeoutError: Timeout while flushing cache: ... /flush_cache
  (Caused by NewConnectionError(... [Errno 111] Connection refused))
```

The py-spy dump puts every rank in `cudaStreamSynchronize` under a host-to-device
copy in `alloc_for_extend` (`allocation.py:355`) while preparing a prefill batch.
That copy is trivial, so the stream was already blocked behind an earlier kernel
that never retired. Four ranks stuck identically points at the engine's own
collective or MoE path rather than a single-rank fault.

This configuration is the first to exercise `--rollout-mxfp8`, so
`flashinfer_trtllm_routed` and `flashinfer_cutlass` are the new elements. The
node returned to `idle` with no drain reason, and the engine had served roughly
fifteen minutes of traffic before hanging, so a transient has not been ruled out.
Rerunning is the cheapest way to tell a flake from a systematic hang.

### 424651 / 424808 / 424849 — phase 1 backend survey

424651 reproduced 424444's hang on a different node, so it is systematic rather
than a node fault. Its timeline places the hang precisely:

| Time | Event |
|---|---|
| 11:46:18 | first `release_memory_occupation` — step 0 rollout done |
| 11:48:31 | `resume_memory_occupation` — training done |
| 11:48:33–11:49:27 | `update_weights` — first online weight sync |
| ~11:50:36 | scheduler's last progress |
| 11:55:36 | watchdog fires |

Step 0's rollout ran roughly ten minutes on freshly loaded weights without
trouble. The hang lands about a minute into the rollout that follows the first
online weight update, which points at the update path rather than steady-state
MoE kernels.

The remaining two runs surveyed alternative MoE runners, and both fail before
generating a token:

| Backend | Outcome |
|---|---|
| `auto` (→ Triton) | `assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]` — the kernel wants `[128, 128]` block scales, MXFP8 groups 32 |
| `deep_gemm` | JIT build fails on `static_assert(kGroupSize == 128)` |
| `flashinfer_trtllm_routed` | the only one that runs; hangs after the first weight update |

So MXFP8 rollout has exactly one working MoE backend on SM100, and stabilizing it
is what phase 1 needs. `SGLANG_MOE_RUNNER_BACKEND` and `SGLANG_FP8_GEMM_BACKEND`
are now launcher inputs so this can be swept without editing the recipe.

The online updater does emit MXFP8: `quantize_params_mxfp8` is selected from the
checkpoint's `quant_method`, and its parameter list matches the conversion tool's
skip list (both keep `wo_a` in BF16).

### 424914 — two-step probe, FAILED (34 m 03 s)

Two rollouts, eval off. Step 0 generated for roughly twenty minutes without
trouble, the first weight sync completed at about 24 minutes, and the watchdog
fired shortly after. That is the fourth reproduction across three nodes, and it
pins the trigger: an MXFP8 rollout survives arbitrarily long on the weights it
loaded, and wedges on the rollout that follows an online weight update.

`mxfp8_interleave_sf.py` looked like a candidate — a post-load scale layout the
updater might bypass — but it interleaves KV-cache scale factors, not weights,
so it does not apply.

Phase 1 is blocked here. The remaining suspects are inside the FlashInfer
TRT-LLM MoE kernel's handling of scales written by the online updater, which
needs kernel-level work rather than another configuration sweep.

Phase 2 does not share this code path: it serves the release checkpoint through
`flashinfer_mxfp4`, so it is worth running independently rather than queueing
behind phase 1.

### 425303 — phase 2, FAILED (23 m 39 s)

`SGLANG_DSV4_FP4_EXPERTS=1` and `moe_runner_backend=flashinfer_mxfp4` resolved as
intended and the engines loaded the release checkpoint unmodified. The run
reached the end of step 0 and died there: the trainer's weight-update POSTs came
back `ConnectionError`, so an engine was already gone before the first sync, and
no watchdog had fired.

Both phases therefore fail in the same window — the first rollout completes, the
handover into training does not — while the immediate symptom differs by backend.
That the two independent MoE paths share a failure window makes the rollout
engines' offload/update handshake the more likely locus than either kernel.

### 425993 — phase 1 unblocked, COMPLETED (1 h 03 m)

Two full steps of MXFP8 training against MXFP8 rollout, clean exit. The change
that unblocked it is a post-update validation pass: `end_weight_update` now takes
`check_weights`, driven by `--check-weight-update-equal`, and asks each engine
for a checksum over its finalized tensors before generation resumes.

Every engine returned `success: True` with matching checksums, so the weights the
update installs are correct — the earlier hangs were not corruption. What the
check adds is a full read of every tensor, and with it the run clears the point
where five previous attempts wedged. The likely reading is a race between the
update's kernel-layout finalization and generation resuming, which the read
serializes; that is inference from the timing, not something the checksums prove.

The MXFP4 quantizer gap in phase 2 is unaffected by this and still stands.

### 429159 — phase 2, FAILED (23 m 35 s)

Died at the initial `update_weights()` that runs before the first rollout, so
nothing was measured. Every engine's four TP ranks raised
`AttributeError: 'Parameter' object has no attribute 'weight_loader'` inside
`load_weights` and the servers SIGQUIT together; the trainer saw only a closed
connection. Ran commit `4324d6113`, whose `uint8` scale convention was itself
wrong — see the phase 2 defects below.

### 430045 — phase 2 smoke, FAILED (23 m 23 s), commit `3f4600b6a`

Never reached the weight update. Both patched SGLang sources verified, then
every trainer actor asserted during `init()`:

```
--rematerialize-param-from-master-weight cannot restore 6 params
  (not in the DDP param buffers nor in the extras backup):
  ['module.module.decoder.layers.0.mlp.router.weight', ...]
```

`--moe-router-freeze-gate` keeps the router weights out of the DDP buffers, so
rematerialization cannot cover them. This was a submission error rather than a
finding: the run took the launcher's defaults instead of the environment the
earlier phase 2 run established, which moved six settings at once —
`REMATERIALIZE_PARAM_FROM_MASTER_WEIGHT`, `OFFLOAD_TRAIN_TARGET`, `MODE`,
`USE_FAULT_TOLERANCE`, `SGLANG_MEM_FRACTION_STATIC` and the patch list. The
launcher's defaults are not the configuration any run here uses; pass the
environment explicitly.

### 430381 — phase 2 smoke, commit `fce6ca869`

429159's environment with one change: `patches/mxfp4_trtllm_hot_reload.patch`
stacked after the existing `sglang_tensor_update_cuda_sync.patch`. One rollout,
evaluation off. Tests the three MXFP4 hand-over fixes together.
`--check-weight-update-equal` is deliberately off: the comparison does not model
the FP4 kernel layout and would be expected to report a difference that is not
one.

### 431471 — phase 1 complete, COMPLETED (1 h 28 m), commit `9918cd1d1`

Four steps of MXFP8 training against MXFP8 rollout, clean exit. 427089's
environment with one change: `patches/weight_update_memory_log.patch`.

| Step | `train_rollout_kl` | `train_rollout_logprob_abs_diff` | `perf/rollout_time` |
|---:|---:|---:|---:|
| 0 | 0.007814 | 0.04838 | 232.8 s |
| 1 | 0.008170 | 0.05231 | 233.0 s |
| 2 | 0.007550 | 0.04878 | 237.0 s |
| 3 | 0.007929 | 0.04850 | 238.6 s |

The mismatch oscillates within 0.0076–0.0082 and does not drift over four
steps. Rollout time rises 2.5% across the run.

## Where the memory goes during an update

Driver figures for one engine's rank 0, at the end of each of the five updates:

| Update | `reserved` | driver used | free |
|---:|---:|---:|---:|
| 1 | 172018 | 90878 | 192383 |
| 2 | 172050 | 92914 | 190347 |
| 3 | 172060 | 92394 | 190867 |
| 4 | 172054 | 92304 | 190957 |
| 5 | 172106 | 92946 | 190315 |

The allocator's own figures are not usable here: `memory_allocated` reports
164 GiB against a driver-reported 83 GiB in use, because the memory saver
releases physical pages while keeping the allocator's bookkeeping. Only
`mem_get_info` describes the device.

The first update raises device usage by about 6 GiB and it does not come back,
but subsequent updates settle onto a plateau rather than accumulating, and
`reserved` moves by 88 MiB across the whole run. Host RSS stays near 100 GiB.
So the update path does not leak, and memory accumulation is not what ends
these runs. That is the third cause proposed for the hang and refuted by
measurement.

### What the failing runs actually did

Rollout durations tell the story the watchdog obscures:

| Job | Rollout times | Outcome |
|---|---|---|
| 425993 | 230, 224 | clean exit at 2 steps |
| 427554 | 229, 532 | engine watchdog |
| 427089 | 235, 235, 1001 | engine watchdog |
| 431471 | 233, 233, 237, 239 | clean exit at 4 steps |

A run that fails does not stall: one rollout takes two to four times as long as
its predecessors, and the scheduler's 300 s watchdog fires during it. Phase 2's
430381, whose rollout ran 82 minutes without finishing, is the same shape
further along. The failure is a throughput collapse, not a deadlock, and it did
not reproduce in 431471 — whose only deliberate difference was the
instrumentation. Treat it as intermittent until something reproduces it on
demand.

### 433028 — phase 2 reaches training, COMPLETED (51 m), commit `618c37b7b`

Phase 2's environment with dynamic sampling off, so the rollout finishes in one
wave instead of resampling. Both weight updates completed — 65.4 s and 60.0 s —
so the hand-over works. What it serves does not:

| Metric | 433028 (MXFP4 rollout) | 431471 (MXFP8 rollout) |
|---|---:|---:|
| `rollout/raw_reward` | 0.0 | 0.51–0.59 |
| `rollout/truncated_ratio` | 0.996 | — |
| `rollout/response_lengths` | 4087 | — |
| `train_rollout_kl` | 0.3347 | 0.0078 |
| `train_rollout_logprob_abs_diff` | 0.4502 | 0.0484 |

Nothing is answered correctly, and 99.6% of responses run to the 4096-token
limit at a mean length of 4087 — the model does not stop. A mismatch 43 times
phase 1's is not a quantization gap; the rollout is serving wrong weights.

This also explains 430381's 82-minute rollout. With dynamic sampling on, every
group's rewards agree at zero, so every group is discarded and sampling never
converges. The rollout was not slow: it was being thrown away.

The defect is not in the encoder or in the layout restore. `mxfp4_quantize`
reproduces real checkpoint tensors byte for byte
(`validate_mxfp4_quantize.py`), and an update reproduces the initial load byte
for byte (`validate_mxfp4_hot_reload.py`). But that second check stubs out
`Fp8MoEMethod.process_weights_after_loading`, which the real path runs *first*,
before the reorder and shuffle. Whether that pass is reentrant across a restore
is untested, and is the next thing to measure.

### 440119 — the weights phase 2 serves are correct

Phase 2 with `--check-weight-update-equal --check-weight-update-allow-quant-error`
and no skip list. The checker snapshots the engine's freshly loaded weights,
poisons them with random values, lets the update run, and compares.

**All 1352 tensors compared equal. Not one `max_abs_err` line was emitted.**

The checker skips some tensors silently — `_is_skip_weight_check` drops kv-cache
scales and post-load placeholders — so the pass is only worth what it covers.
Auditing the reported list: 1442 names on one rank, of which 619 attention, 301
routed expert, 86 shared expert, 86 dense, and the rest buffers. The four
kernel-layout parameters appear for every one of the 43 layers,
`mlp.experts.{w13_weight, w2_weight, w13_weight_scale_inv, w2_weight_scale_inv}`,
172 entries in all. Nothing relevant was skipped, so the pass is real.

The run then reproduced the failure it was sent to explain, on weights it had
just certified:

| | 440119 | 433028 |
|---|---:|---:|
| `rollout/raw_reward` | 0.0 | 0.0 |
| `rollout/truncated_ratio` | 0.988 | 0.996 |
| `train_rollout_kl` | 0.3258 | 0.3347 |

That covers attention, the shared expert, the dense layers *and* the routed
experts — the MXFP4 payloads compare raw and bitwise, because
`select_comparable_weight` only recognizes `Fp8LinearMethod` and `Fp8MoEMethod`,
and `Mxfp4FlashinferTrtllmMoEMethod` wraps rather than subclasses them.

So both remaining candidates die at once. The hand-over is correct, in the
MXFP4 expert path and the block-scaled FP8 path alike, and phase 2's
`raw_reward` of 0.0 with 99.6% truncation is not a weight error. Every
hypothesis this bring-up has proposed for phase 2 has now been refuted by
measurement.

What that leaves is the compute side rather than the weight side: the TRT-LLM
MXFP4 kernel itself, the MXFP8 activation quantization
`flashinfer_mxfp4_moe_precision=default` performs on its inputs, or routing and
top-k under the FP4 branch. None of these is exercised by phase 1, and none is
touched by a weight comparison.

One limit worth stating: the check compares the updated weights against the
engine's *own* initial load. It cannot see an error that is already present in
that load. SGLang serving this checkpoint normally is well-trodden, so this is
unlikely, but it is not excluded by this run.

### 443944 / 444087 — the serving path is fine, and why that settles it

The release checkpoint served on one node, stock SGLang, packed MXFP4 experts on
`flashinfer_mxfp4`, no patch and no weight update:

| Prompt | Output | Tokens | Finish |
|---|---|---:|---|
| `2 + 2` | `…</think>4` | 26 | stop |
| capital of France | `…</think>Paris` | 45 | stop |
| `17 * 3` | `…</think>51` | 22 | stop |
| complete "the sky is" | coherent | 124 | stop |

None ran to the limit. 443944 failed first because the checkpoint ships no chat
template; the smoke now renders through DeepSeek's own encoder, as the rollout
does.

Three facts now hold at once: the serving path works, the weights after an
update compare equal, and serving *after* an update is broken. Together they
point at state the kernel reads that `named_parameters()` does not describe —
and the CUDA graphs SGLang captures at startup record the *addresses* of the
tensors they read.

Measured directly: every expert parameter moved across an update.

```
w13_weight             MOVED  0x73be5d00f800 -> 0x73be5d010400
w2_weight              MOVED  0x73be5d013c00 -> 0x73be5d014400
w13_weight_scale_inv   MOVED  0x73be5d013800 -> 0x73be5d00ae00
w2_weight_scale_inv    MOVED  0x73be5d015c00 -> 0x73be5d016400
```

So the graph replays against the previous weights while every Python-visible
view holds the new ones — which is exactly a weight checker that passes and a
model that answers nothing. The MXFP8 path avoids this: `fp8.py`'s
`_copy_or_rebind` prefers an in-place copy, and its comment says so. The MXFP4
patch adopted that helper's intent for loader attributes and missed its intent
for addresses.

The fix keeps the kernel-layout storage and copies each rebuild into it.
`validate_mxfp4_hot_reload.py` now asserts the addresses hold, and they do.

### 454574 — the address fix changes nothing

| | 454574 (addresses stable) | 440119 | 433028 |
|---|---:|---:|---:|
| `rollout/raw_reward` | 0.0 | 0.0 | 0.0 |
| `rollout/truncated_ratio` | 1.000 | 0.988 | 0.996 |
| `train_rollout_kl` | 0.3533 | 0.3258 | 0.3347 |

The instability was real and the fix does what it claims, but it is not the
cause. Add it to the list of measured refutations rather than treating it as
progress.

What the reasoning missed is that the control and the failing runs differ by
more than a weight update. The one-node smoke served **stock** SGLang; every
phase 2 run carries three patches. If one of those breaks the *initial* load,
the model is wrong from its first rollout and "it only breaks after an update"
was never established.

### 456107 — the patches do not change the initial load

The same one-node smoke, carrying all three patches, no weight update: identical
outputs to the unpatched 444087, the same token counts (26, 45, 22, 124), 0/4
truncated. Locally the same holds at finer grain — building the kernel layout
from identical inputs gives byte-identical results with and without the patch,
on all four parameters.

So the patches are exonerated, and four things now hold together:

- serving works, patched and unpatched;
- the weights after an update compare equal, all 1352 of them;
- the kernel-layout addresses now hold across an update;
- serving after an update still answers nothing.

Whatever breaks is rebuilt by the update, is not a parameter, and is not an
address.

### 456154 / 456194 / 456224 — the update itself does not break it

`update_smoke.sbatch` serves the checkpoint on one node, hands the model back
the tensors it was loaded from, and generates on both sides. An identity update
should be invisible, so a degradation across it would reach the whole failure in
twenty-three minutes on four GPUs.

| Run | What it added | Result |
|---|---|---|
| 456154 | identity update over every tensor | asserted, see below |
| 456194 | routed experts only | 0/4 truncated before and after |
| 456224 | plus the engine's release/resume cycle | 0/4 truncated before and after |

Correction after the root-cause run: this harness had
`enable_memory_saver=False`. The API calls in 456224 returned successfully but
did not release or restore physical memory, so this row did not cover the
production memory cycle.

456154 failed on `model.layers.0.self_attn.wo_b.weight_scale_inv`:
`assert self.data.shape == loaded_weight.shape`. The first load reshapes the
attention scales, so their own checkpoint bytes no longer fit the parameter they
came from. That is the same second-load problem in stock SGLang, on a path this
work does not touch; phase 2's real update does not hit it because miles sends
scales requantized to the shape the parameter now has.

So the MoE update path carrying correct data is fine; the real memory cycle was
still untested here. What remained untested between the reproducer and phase 2 was the
transport — phase 2 delivers weights as flattened buckets over CUDA IPC, not as
plain tensors — and the non-expert parameters, which phase 2 also updates and
the reproducer cannot send from the checkpoint.

### The first transport variants did not get a verdict

Five further runs (456275, 456340, 456409, 456493, 456568, 456670) tried to send
the identity update over flattened buckets and CUDA IPC. Every one died in the
harness rather than in the system under test: a whole shard per bucket did not
fit, then 256 MiB buckets did not, then the allocator held each bucket after
sending, then closing the update window every four shards still reached only
shard 32 of 48, and one run asserted because `release_memory_occupation`
requires an idle server that `generate` had not yet drained.

The cause is structural. The reproducer builds its buckets in the same process
that hosts a TP worker, so exporter and importer compete for one GPU; the
rollout never does this, because trainer and engine are separate processes that
negotiate their split. Tuning around that consumed seven runs and produced no
experimental result, which is the same mistake as before in a new place: a probe
that only runs at full scale needs a full-scale run to expose each of its own
defects.

One real observation survives, independent of whether the reproducer ever runs:
**imported IPC buffers accumulate for the lifetime of a begin/end window and are
released only when it closes.** With 64 MiB buckets the run reached four times
as many shards as with 256 MiB, so what accumulates is the total bytes in flight,
not the number of handles. This also corrects the reading of the earlier
configuration sweep, which took "smaller buckets, fewer steps" as evidence of
per-handle accumulation; the memory does not accumulate per handle.

Next, and not by tuning this further: either give the reproducer its own GPU for
staging, or go back to the eight-node path with a specific diagnostic rather than
a general one.

### 2026-08-17 — dedicated-exporter B300 controls pass

The local node has eight B300s, so the reproducer can use GPU 0-3 for TP4/EP4
and a separate GPU for CUDA-IPC staging. The official release, exact OCI image
digest and both runtime patches were used. Unlike 459395, the strict run keeps a
single begin/end transaction across all 48 shards. It also retries
`flush_cache()` until the scheduler explicitly reports idle; the first attempt
was in fact too early and the second succeeded.

The strict identity run sent 70,656 routed-expert tensors in 64 MiB flattened
buckets. It completed with `0/4` prompts truncated before and after, identical
text and exit code zero. A two-shard window had produced `4/4` truncation, but
that is a false positive caused by postprocessing a partially loaded model at
every close, not the transaction Miles uses.

Two changed-payload round trips then tested what identity cannot:

| Payload mutation | Mutated model | After restoring release payload |
|---|---|---|
| flip both E2M1 sign bits in every packed expert weight byte | `4/4` truncated, repeated garbage | `0/4`, answers `4`, `Paris`, `51`, `blue` |
| increment every expert UE8M0 scale exponent by one | immediate incoherent text | `0/4`, full text identical to the initial baseline |

The small hot-reload contract was strengthened in parallel: its second load now
uses different random packed weights and scales and compares the rebuilt layout
against a fresh load of that new payload. All four parameters match byte for
byte while retaining their original addresses.

So the kernel reads changed weights and changed scales, and restoring either is
reversible. Packed MXFP4 reload, flattened CUDA-IPC and TP4/EP4 work locally on
B300. These historical runs also had memory saver disabled, so they did not
cover the actual release/resume cycle. The remaining difference was no longer the
transport-format cell. The local control sends 70,656 checkpoint-origin HF
expert tensors in 64 MiB chunks. Miles exports BF16 trainer tensors through
Megatron Bridge, re-quantizes them, includes non-expert weights and chunks at
1 GiB. At this point, replaying that full scope was the next discriminating
measurement. The `miles_full` result at the top of this file supersedes that
recommendation: its 1 GiB full-model transaction passes both the strict checker
and generation.

### 456794 — CUDA graph replay is not the mechanism either

The one-node control served four prompts; the rollout serves 256 at a time, so
the two land in different capture buckets and the control may never have
exercised the replay path the rollout uses. Disabling graphs outright settles
it:

| | 456794 (no graphs) | 454574 | 440119 |
|---|---:|---:|---:|
| `rollout/raw_reward` | 0.0 | 0.0 | 0.0 |
| `rollout/truncated_ratio` | 0.996 | 1.000 | 0.988 |
| `train_rollout_kl` | 0.4319 | 0.3533 | 0.3258 |
| `perf/rollout_time` | 959.8 s | — | — |

Rollout takes twice as long without graphs, as expected, and the model is just
as broken. Twelve hypotheses have now been refuted by measurement, three of them
proposed and implemented by this work.

### 457348 — dequantizing the experts fixes everything

`SGLANG_DSV4_FP4_DEQUANT=1` unpacks the release checkpoint's routed experts into
FP8 during load, so every parameter takes the block-scaled FP8 hand-over and none
of the MXFP4 work participates. `moe_runner_backend` must be `auto`
(`fp8.py:366` asserts it), and the updater has to send FP8 rather than packed
payloads — 457280 died on exactly that, `size of tensor a (4096) must match
tensor b (2048)`, the packing factor.

| | 457348 (dequant) | broken phase 2 | phase 1 |
|---|---:|---:|---:|
| `rollout/raw_reward` | **0.742** | 0.0 | 0.51–0.59 |
| `rollout/truncated_ratio` | **0.258** | 0.988–1.000 | 0.258 (0731 baseline) |
| `rollout/response_lengths` | 2270 | 4087 | — |
| `train_rollout_kl` | **0.00560** | 0.326–0.432 | 0.00755–0.00817 |
| `train_rollout_logprob_abs_diff` | 0.0398 | 0.450 | 0.0484 |
| `perf/rollout_time` | 64.8 s | — | 233 s |

Every number is healthy, and the mismatch is *lower* than phase 1's. So the
block-scaled FP8 hand-over is sound, and the defect is in the MXFP4 expert path.

That collides with the weight check head-on. After an MXFP4 update all 1442
tensors compare equal to a fresh load, on an audited list that includes all four
kernel-layout parameters for every layer — and the model still answers nothing.
Both cannot be explained by the weight values, so what differs has to be
something the MXFP4 path establishes on its first `process_weights_after_loading`
and does not re-establish on the next: state outside `named_parameters()` that
the FP8 path either does not have or rebuilds correctly.

### 458663 — phase 2 completes four steps, COMPLETED (1 h 04 m)

Same configuration as 457348 with `NUM_ROLLOUT=4`.

| Step | `train_rollout_kl` | `logprob_abs_diff` | `raw_reward` | `truncated_ratio` | `rollout_time` |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.005688 | 0.03980 | 0.746 | 0.289 | 66.6 s |
| 1 | 0.005507 | 0.04129 | 0.621 | 0.461 | 70.4 s |
| 2 | 0.005424 | 0.04171 | 0.625 | 0.492 | 69.0 s |
| 3 | 0.005045 | 0.03938 | 0.703 | 0.418 | 54.7 s |

## The comparison

Both phases train the same MXFP8 checkpoint for four steps and differ only in
what the rollout serves.

| | Phase 1 (431471) | Phase 2 (458663) |
|---|---|---|
| Rollout checkpoint | MXFP8, converted from the BF16 cast | the release checkpoint as shipped |
| Routed experts | MXFP8, group 32 | packed MXFP4, unpacked to FP8 at load |
| Everything else | MXFP8, group 32 | block-scaled FP8, `[128, 128]` |
| `train_rollout_kl` | 0.00755 – 0.00817 | **0.00505 – 0.00569** |
| `logprob_abs_diff` | 0.0484 – 0.0523 | **0.0394 – 0.0417** |
| `raw_reward` | 0.512 – 0.590 | 0.621 – 0.746 |
| `rollout_time` | 233 – 239 s | 54.7 – 70.4 s |
| Drift over four steps | none | none |

An independent reference exists for the setup itself: PR #1340, which added
`--train-mxfp8`/`--rollout-mxfp8` in July, reported five steps at
`kl` 0.01099–0.01120 and `abs_diff` 0.0897–0.0905 on the checkpoint that
predates the 0731 release. All three sit in the same range and none drifts, and
they order the way the formats do — 0.0110 for the earlier checkpoint, 0.0078
serving a converted MXFP8 copy of the release, 0.0055 serving the release's own
weights.

Phase 2's mismatch is about a third lower than phase 1's, and neither drifts.
Serving the release checkpoint's own weights agrees with the trainer *better*
than serving a checkpoint converted from the same BF16 cast — the conversion to
MXFP8 costs more agreement than the release quantization does. Rollout is also
three to four times faster, though that reflects the MoE runner as much as the
weight format.

One caveat this does not measure: phase 2 here serves the release experts
**unpacked to FP8**, not packed MXFP4. That is the configuration that works; the
packed path remains broken and is the open item below.

And the bisection that reached it moved two things at once, which is worth
stating plainly rather than reading past: unpacking the experts also forces
`moe_runner_backend` from `flashinfer_mxfp4` to `auto`. So it separates the
working configuration from the broken one, but it does not separate the *weights*
from the *kernel*. Two facts point at the kernel rather than the payload: serving
the packed checkpoint untouched works (444087, 456107), and an identity update of
the packed experts on one node, with memory saver disabled, leaves generation
unchanged (456194, 456224). What no run has yet isolated is packed weights served
by a different MXFP4 kernel — `marlin` and `humming` both accept them
(`fp8.py:371`, `fp8.py:378`).

Neither substitute is usable, and checking that cost a run it should not have.
459149 asked for `marlin` and spent 45 minutes reaching
`RuntimeError: MXFP4 Marlin requires SM90 or SM120` — GB200 is SM100, and one
minute of reading in the image would have said so. `humming` has no arch guard
but also no `restore_load_layout`, so a second load would write into the layout
its `process_weights_after_loading` already transformed — the same defect this
work fixed for the FlashInfer path, which makes it a confounded control rather
than a clean one.

So the kernel and the payload cannot be separated by swapping backends on this
hardware.

Expert parallelism looked like the remaining axis — a correct per-rank weight
comparison and a wrong global result are not contradictory if experts land on
the wrong rank — but the logs refute it for free: the failing 454574 and the
working 458663 both run `tp_size=4, ep_size=4`, the same topology the one-node
reproducer uses.

At the time of the OCI runs, crossing off what the working run and the
reproducer between them already covered left one untested cell:

| | packed MXFP4 | unpacked FP8 |
|---|---|---|
| plain tensor hand-off | 456194, 456224 — pass | — |
| flattened bucket over IPC | local B300, 2026-08-17 — pass | 458663 — pass |

The combination of bucket transport with packed MXFP4 experts was the one thing
only the failing runs did when this table was first written. It is no longer an
untested cell: dedicated-staging expert controls complete identity,
changed-weight and changed-scale round trips, and the later full-model control
also completes with rank-local buffers on GPU 0-3. The old 459395 route was
blocked by its single shared staging allocation strategy, not by CUDA IPC or
the update topology itself.

The obvious mechanism for such a pairing — reconstructed views outliving the
buffer they point into — is already excluded. The engine-side patch drops each
imported tensor and calls `gc.collect()` and `torch.cuda.ipc_collect()` per
bucket, the trainer releases `long_live_tensors` before `update_weights()`
returns, and the comparison runs after that. It compares equal.

### 460099 — what the broken rollout actually generates

Every judgement of "broken" so far came from aggregates. `--save-debug-rollout-data`
keeps the samples, and `read_rollout_dump.py` reads them. The shape settles two
questions that no aggregate could.

```
[0] reward=0  4284 tokens   'equal' x3908 of 4069 (96%)
    HEAD  " about the solution for a form $AD$ and $AB $ -E$ and $A can is the
           answer to the problem. In the point $ = the $ = and the answer to ..."
    TAIL  "equal equal equal equal equal equal equal equal equal ..."
[1] reward=0  4284 tokens   'G.' x1603 of 2065 (78%)
[2] reward=0  4284 tokens   "$_$_$_$_$_ ..."
```

The text is not fluent, so a stop condition or a chat template is not the cause
— the model really is producing wrong values. But it is not noise either: the
prompt's own vocabulary survives and stays roughly in position (`$ABCD$`, `$AB$`,
`$AD$`, "area", "line", "Answer"), and the failure is that nothing composes,
ending in single-token collapse. Locally plausible, globally incoherent is what a
wrong feed-forward contribution looks like on top of working attention — which
matches attention taking the block-scaled path that 458663 proves sound.

The same dump carries `rollout_routed_experts`, so the router can be checked
directly rather than inferred:

```
[0] 1105014 picks over 256 experts, top5 covering 13%
[1] 1105014 picks over 256 experts, top5 covering 12%
```

Every expert is used and the top five take an eighth of the traffic, against
about a fiftieth for a perfectly flat distribution. That is ordinary mild
concentration, not collapse. **Routing is healthy; what the experts return is
wrong.**

Dtype was the last thing the kernel could dispatch on that a byte comparison
would miss — `fp8.py:1560` and the rebuild both re-view without changing bytes.
Driving a reload on a two-expert layer and comparing dtype and shape rather than
only bytes finds all four parameters identical: `uint8`, `uint8`,
`float8_e4m3fn`, `float8_e4m3fn`, zero mismatches.

So the contradiction is now as sharp as the instruments can make it. The served
bytes are identical to a fresh load, on an audited list, across all 32 ranks.
A fresh load with those bytes answers correctly. The same bytes after an update
answer nothing. Topology, transport, payload and non-expert path are each shared
with a run that works. Whatever remains is not visible to a weight comparison,
not a tensor address, not a CUDA graph, and not the serving path in isolation.

### What phase 2 changes besides the experts

The two rollout checkpoints do not share a hand-over path:

| | quant_method | weight_block_size |
|---|---|---|
| `DeepSeek-V4-Flash-0731-MXFP8` (phase 1) | `mxfp8` | `[1, 32]` |
| `DeepSeek-V4-Flash-0731` (phase 2) | `fp8` | `[128, 128]` |

Phase 1 sends every parameter through `quantize_params_mxfp8`. Phase 2 sends
the routed experts through the MXFP4 processor and *everything else* — attention,
the shared expert, the dense layers — through block-scaled FP8 at `[128, 128]`,
which no run in this experiment has ever exercised. Broken attention produces
the same signature as broken experts: nothing answered, everything truncated.

So the fault is not necessarily in the MXFP4 work. The cheapest way to tell is a
bisection: serve the release checkpoint with `SGLANG_DSV4_FP4_EXPERTS=0` and
`SGLANG_DSV4_FP4_DEQUANT=1`, which dequantizes the packed experts at load and
puts every parameter on the block-scaled FP8 path. If that is also broken the
MXFP4 code is exonerated and the block-scaled path is the suspect; if it works,
the fault is in the expert hand-over after all.

Ruled out so far, each by measurement rather than reading: the encoder
(`validate_mxfp4_quantize.py`), the layout restore
(`validate_mxfp4_hot_reload.py`), the FP8 base method's post-load pass — its
FP4 branch only re-views the payload as `int8` and does not touch scales — the
scale suffix, which matches the FP8 path phase 1 uses, and the loader's write
semantics, which narrow and `copy_` exactly as the verified round trip assumed.

### Where the watchdog actually fires

427089's py-spy dump, taken by the watchdog itself, puts the scheduler at
`get_next_batch_to_run` line 2802 — the call into `get_new_batch_prefill` —
with native frames below it ending in `clock_nanosleep`. The only `time.sleep`
in `scheduler.py` is an init-time test hook, so that is a native wait: a CUDA
synchronization, not Python.

424444's dump agrees from the other side: `cudaStreamSynchronize` beneath
`alloc_for_extend`, also in prefill. Two independent failures put the stall in
the same place — the first prefill after a weight update, blocked on the device.

So the engine is not busy and not out of memory. It is waiting on a kernel that
never retires, which is why every configuration sweep that changed scheduling
pressure moved the step count around without ever fixing it.

### Configuration sweep against the hang

Steps completed before an engine tripped its watchdog, all on phase 1:

| Configuration | Steps |
|---|---:|
| baseline | 0 |
| `--check-weight-update-equal` | 2 |
| above, `ENABLE_EVAL=0` | 3 |
| above, plus an explicit device sync after `postprocess_weight` | 1 |
| above, `mem-fraction 0.5`, 64 MiB update buckets | 1 |
| above, 1 GiB update buckets | in flight as job 427554 |

Two root causes were proposed and both were refuted by these runs. The MoE
shuffle *is* re-applied — `end_weight_update` calls `postprocess_weight`, and the
post-update checksums come back consistent across engines. A race between that
finalization and resuming generation is not it either: forcing a synchronize made
the run fail sooner, not later.

What the sweep does show is a monotone response to load, and a sharp regression
when the update buckets shrink — four times as many IPC transfers per sync cost
two steps. That points at per-handle accumulation in the CUDA IPC path rather
than a race, which is what the 1 GiB run tests.

Reading engine-side `torch.cuda.memory_allocated`/`reserved` and a live IPC
handle count around each update would settle it, and is cheaper than continuing
to sweep configurations.

## Mismatch measurements

Phase 1's numbers are in hand and reproduce across three independent runs. These
are the baseline phase 2 is meant to be compared against.

| Job | Step 0 `train_rollout_kl` / `logprob_abs_diff` | Step 1 |
|---|---|---|
| 427089 | 0.00800 / 0.0505 | 0.00894 / 0.0546 |
| 425993 | 0.00760 / 0.0472 | 0.00851 / 0.0522 |
| 427554 | 0.00780 / 0.0472 | — |

Step 0 agrees to about 7% across runs, so the measurement is stable. Note that
the metric is produced during training on the rollout just taken, which makes it
available after a single step: comparing rollout formats does not need the
four-step run, only a run that reaches training.

## What blocks each phase

**Phase 1 is not blocked on correctness.** Its weight-update path is sound. The
kernel layout the MXFP8 experts are served in preserves both the dtype and the
extent of the layout weights load in, so each update refills the parameters from
scratch and the layout is rebuilt from canonical data. Rebuilding is not
idempotent — calling it twice on its own output changes all four parameters —
but it is never fed its own output. What remains is the hang described in the
sweep above, which costs steps rather than correctness.

**Phase 2 was blocked by three defects in the MXFP4 rollout path**, all in how
weights are handed over rather than in the encoding, and all fixed and checked
against real tensors before submission:

1. Building the kernel layout replaced the expert parameters with plain ones,
   dropping the loader attributes `create_weights` installs. The second load —
   the first online update — raised `AttributeError: 'Parameter' object has no
   attribute 'weight_loader'`. This is what killed 429159.
2. Unlike the MXFP8 path, the MXFP4 kernel layout differs from the load layout in
   dtype, and for the second-gemm scale in extent. Writing weights back into it
   casts them to the wrong values instead of failing: a scale byte of 130 stored
   into the `float8_e4m3fn` parameter reads back as 112. The layout is now
   restored before weights arrive, each parameter released before its replacement
   is allocated so both layouts are never resident.
3. MXFP4 scales are staged in a float parameter and converted back to UE8M0 once
   the tensor has arrived, so a scale has to carry its power of two rather than
   the byte encoding it. Handing over `uint8` collapsed the exponents 120, 127,
   130 and 140 onto a single byte; `float8_e8m0fnu` round-trips exactly.

End to end, an update now reproduces the initial load byte for byte on all four
expert parameters. The earlier claim that phase 1 fails because the MoE shuffle
is not re-applied, and that phase 2 fails only for want of an MXFP4 quantizer,
were both wrong; the quantizer was necessary but not sufficient.

## Environment

- Cluster `oci-aga`, account `coreai_devtech_all`, partition `batch`, QoS
  `short` (2 h). The `interactive` QoS requires a reservation and is not usable
  for batch submission.
- Workspace
  `/scratch/fsw/portfolios/coreai/projects/coreai_devtech_all/users/lbo/dsv4_flash_rl`.
- Image `radixark/miles:latest`,
  digest `sha256:08be00658cd24eaa364ca4ad0b1a3911dfbe4adc04fd0c148e4241402fb40812`.
  Each job re-imports it through pyxis, which costs several minutes; caching a
  `.sqsh` on the shared filesystem would remove that.
- The same image and a B300 host are available locally, so conversion and kernel
  work can be validated before any job is submitted.
