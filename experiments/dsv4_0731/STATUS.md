# DeepSeek-V4-Flash-0731 MXFP8 RL bring-up

Last updated: 2026-08-12

## Objective

Two phases, both training with the Megatron MXFP8 recipe on 32 GB300 GPUs:

1. **MXFP8 train + MXFP8 rollout.** Rollout serves an MXFP8 checkpoint derived
   from the release, so both sides use the same nominal format.
2. **MXFP8 train + native rollout.** Rollout serves `DeepSeek-V4-Flash-0731`
   as shipped — MXFP4 routed experts against MXFP8 activations. The
   train/rollout mismatch is expected to rise relative to phase 1.

Both phases share one trainer checkpoint, so the mismatch delta isolates the
rollout-side weight format.

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
