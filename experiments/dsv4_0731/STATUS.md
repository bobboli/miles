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

456154 failed on `model.layers.0.self_attn.wo_b.weight_scale_inv`:
`assert self.data.shape == loaded_weight.shape`. The first load reshapes the
attention scales, so their own checkpoint bytes no longer fit the parameter they
came from. That is the same second-load problem in stock SGLang, on a path this
work does not touch; phase 2's real update does not hit it because miles sends
scales requantized to the shape the parameter now has.

So the MoE update path carrying correct data is fine, and so is the memory cycle
around it. What remains untested between the reproducer and phase 2 is the
transport — phase 2 delivers weights as flattened buckets over CUDA IPC, not as
plain tensors — and the non-expert parameters, which phase 2 also updates and
the reproducer cannot send from the checkpoint.

### The transport variant did not get a verdict

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
