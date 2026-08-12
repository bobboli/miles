# DeepSeek-V4-Flash-0731 MXFP8 RL bring-up

Last updated: 2026-08-11

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

### Why both phases fail at the handover

Each phase hits a distinct gap in the online weight-update path, and both gaps
are invisible until the first update because step 0 runs on weights that went
through `process_weights_after_loading`.

**Phase 1 — MoE expert weights are shuffled at load and not re-shuffled after an
update.** `moe_runner/flashinfer_trtllm.py` runs `shuffle_matrix_a` over the
expert weights and serves them with `is_sf_swizzled_layout=True`. The dense path
in `fp8.py` keeps a separate `weight_scale_inv_swizzled` with the comment "so
store swizzled scales separately to keep weight update working"; the MoE path has
no such provision. An update writes unshuffled weights into buffers the kernel
reads as shuffled, and the next rollout wedges the GPU. This also explains why
the survey found no working alternative: the only MXFP8-capable MoE backend on
SM100 is the one that shuffles.

**Phase 2 — there is no MXFP4 quantizer in the update path.** `quantize_params`
dispatches on the rollout checkpoint's `quant_method`, and the release config
reports `fp8` with `weight_block_size [128, 128]` even though its routed experts
are packed MXFP4. The updater therefore emits block-scaled FP8 into expert
buffers that `SGLANG_DSV4_FP4_EXPERTS=1` created as packed MXFP4, and the engine
dies outright. `processors/` holds fp8, mxfp8, nvfp4 and compressed-tensors
quantizers, but no mxfp4.

Neither is a configuration problem, so no further parameter sweep will clear
them. Phase 1 needs the MoE shuffle re-applied after each update — either by
keeping an unshuffled copy the way the dense path does, or by re-running the
expert post-processing on update. Phase 2 additionally needs an MXFP4 quantizer;
the encoding is fully pinned down by `mxfp4_dequant` and its tests, so writing
the inverse is tractable.

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
