# Handoff: 0731 MXFP8 RL, 2026-08-14

Written for whoever picks this up next. It says where the work stands, what is
open, and what it costs to be wrong about each piece. It does not repeat
[STATUS.md](STATUS.md) — that is the evidence file, and every claim here points
into it. Read [INDEX.md](../INDEX.md) first if you have not.

## Where it stands

Both phases have their four-step numbers. That was the question the experiment
was set up to answer, and it is answered.

| | Phase 1 (job 431471) | Phase 2 (job 458663) |
|---|---|---|
| Rollout serves | MXFP8 converted from the BF16 cast | the release checkpoint, experts unpacked to FP8 at load |
| `train_rollout_kl` | 0.00755 – 0.00817 | 0.00505 – 0.00569 |
| `logprob_abs_diff` | 0.0484 – 0.0523 | 0.0394 – 0.0417 |
| `raw_reward` | 0.512 – 0.590 | 0.621 – 0.746 |
| `rollout_time` | 233 – 239 s | 54.7 – 70.4 s |
| Drift over four steps | none | none |

**The result is the opposite of the premise.** The experiment expected phase 2's
mismatch to be higher, since it serves a coarser weight format. It is about a
third lower. Serving the release's own weights agrees with the trainer better
than serving an MXFP8 copy converted from the same BF16 cast — the conversion
costs more agreement than the release's quantization does.

PR #1340, which added the `--train-mxfp8`/`--rollout-mxfp8` switches this builds
on, reported 0.01099–0.01120 on the pre-0731 checkpoint. All three sit in the
same range and order with the formats: 0.0110 → 0.0078 → 0.0055 as the rollout
gets closer to the release's own weights. That is the only outside check these
numbers have.

## What is open

**Serving the release's experts packed rather than unpacked still produces
nothing usable after an online weight update.** `raw_reward` 0.0,
`truncated_ratio` 0.99–1.00, `train_rollout_kl` 0.33–0.43, reproduced across
four runs. Phase 2's numbers above avoid this path by unpacking at load.

Fifteen hypotheses have been refuted by measurement, three of them proposed and
implemented by this work. STATUS.md carries each. The failure now reads:

> Routing is healthy — 256 of 256 experts used, top five taking an eighth of the
> traffic. The weights after an update compare equal to a fresh load, byte for
> byte, on an audited list covering all four kernel-layout parameters for every
> layer, across all 32 ranks. Dtype and shape survive. Tensor addresses survive.
> Attention takes the block-scaled path that phase 2 proves sound. Serving those
> same bytes without an update answers correctly. **The MXFP4 expert kernel
> returns wrong values from correct weights, and only after a reload.**

What the broken rollout generates is in STATUS.md: locally plausible, globally
incoherent, collapsing into one repeated token — the prompt's own vocabulary
survives and nothing composes. That is a wrong feed-forward contribution on top
of working attention, and it rules out a stop condition or a chat template.

### The next measurement, and why it has to be that one

Every cheap instrument is exhausted. What has never been done is reading values
*inside* the kernel: run the same batch through the MoE before and after an
update and compare the output tensors. That needs instrumentation in
`Mxfp4FlashinferTrtllmMoEMethod.apply`, which is new code rather than new
configuration — worth agreeing on the approach before writing it.

Two dead ends, so they are not retried:

- **Swapping the MXFP4 kernel** does not isolate weights from kernel here.
  `marlin` needs SM90 or SM120 and GB200 is SM100 — a 45-minute run learned what
  a minute of reading the image would have said. `humming` has no
  `restore_load_layout`, so it fails a second load for the reason this work
  already fixed elsewhere.
- **The one-node reproducer** is structurally blocked. It hosts the exporting
  process and an importing TP worker on one GPU, which the rollout never does,
  and seven runs went into making the harness survive its own memory footprint
  without reaching a verdict. `update_smoke.sbatch` still works for the plain
  transport; the flattened-bucket variant does not.

## Merge requests open for review

All on GitLab, none visible outside NVIDIA. Review order is the table order —
the first two are independent of the MXFP4 work and of each other.

| MR | Contents |
|---|---|
| [lbo/miles!1](https://gitlab-master.nvidia.com/lbo/miles/-/merge_requests/1) | MXFP4 → BF16 decode in `tools/fp8_cast_bf16.py`, plus `miles/utils/mxfp4.py` and 12 CPU-only tests |
| [lbo/miles!2](https://gitlab-master.nvidia.com/lbo/miles/-/merge_requests/2) | Weight-update IPC lifetime and the post-update checksum |
| [lbo/miles!3](https://gitlab-master.nvidia.com/lbo/miles/-/merge_requests/3) | Registering the release and configuring rollout from its actual layout |
| [lbo/miles!4](https://gitlab-master.nvidia.com/lbo/miles/-/merge_requests/4) | This directory: the run log, the tools, the retrospective |
| [lbo/sglang!1](https://gitlab-master.nvidia.com/lbo/sglang/-/merge_requests/1) (draft) | MXFP4 kernel-layout hot reload. Prerequisite for the packed path, not a fix for it — draft on purpose |

The sglang MR targets `sglang-miles`, not `main`: the image's SGLang is
`cb05a44`, which is that branch's head. Megatron needs no change.

Three things to know if you rebase or re-split them. The branch is based on an
older `main` than the fork's, so `arguments.py` and `run_deepseek_v4.py` were
applied with `git apply --3way` rather than checked out wholesale — a wholesale
copy silently reverts 83 and 4 lines of upstream work respectively.
`--http-request-timeout` rode along in the same commit but its consumer is only
in an uncommitted working tree, so it was stripped. And the working tree carries
unrelated modifications that predate this work (`model.py`,
`rematerialize_utils.py`, `train.py`, several tests) — the split was done in a
separate `git worktree` to keep them out.

## Formats, in the terms this work uses them

These came up repeatedly in review and are not written down anywhere else.

**Packed** means two 4-bit elements share one byte, low nibble first. MXFP4
elements are E2M1 — four bits — and no storage dtype is four bits wide, so the
stored tensor's last dimension is half the logical one. That halving is what
`size of tensor a (4096) must match tensor b (2048)` was: the engine had
unpacked, the updater had not.

**Unpacked to FP8** expands each 4-bit value into its own E4M3 byte. Same
values, twice the space, and the ordinary block-scaled FP8 kernels can read it.
`SGLANG_DSV4_FP4_DEQUANT=1` does this at load, and forces
`moe_runner_backend=auto` (`fp8.py:366` asserts it).

**`float8_e8m0fnu`** is a byte that holds only an exponent: `value = 2**(b-127)`,
no sign bit, no mantissa, byte 255 is NaN. It exists so a block scale is a power
of two and dequantization is an exponent add rather than a multiply. Two readings
of the same byte cost this work a defect: `scale.view(torch.uint8)` gives 130,
`scale.float()` gives 8.0, and handing SGLang the former where it wants the
latter collapses exponents 120, 127, 130 and 140 onto one value.

**Telling the two layouts apart** cannot be done by name or dtype — the release
and the preview both call the tensor `...w1.weight` and both store one byte per
storage unit. Only the scale's extent separates them:

| | preview | release |
|---|---|---|
| weight | `F8_E4M3` `[2048, 4096]` | `I8` `[2048, 2048]` |
| scale | `F32` `[16, 32]` | `F8_E8M0` `[2048, 128]` |
| last-dim ratio | 128 → blockwise FP8 | 16 → packed MXFP4, group 32 |

`is_mxfp4_weight` tests exactly that ratio. Feeding a packed tensor to the
block-scaled path raises nothing: shapes stay self-consistent and it emits a
plausible tensor of garbage.

**Choosing the block exponent.** `e = ceil(log2(amax / 6))` is the smallest
integer with `6 * 2**e >= amax`, which fills the E2M1 range without clipping.
The `clamp(min=-127)` is not defensive padding: E8M0 stores `e + 127` in a byte,
so an exponent below the floor wraps rather than saturating, and a block whose
largest magnitude is 1e-40 would encode 2\*\*-135 as byte 248, which reads back
as 2\*\*121. There is deliberately no upper clamp — bf16 and fp32 both top out
at `e = 126`, one below the ceiling, so it is unreachable for finite input. This
matches FlashInfer's kernel exactly: `vecMax * (1/6.0f)` then
`__nv_cvt_float_to_e8m0(..., __NV_SATFINITE, cudaRoundPosInf)`, where round-toward-
+inf on a power-of-two format *is* `ceil(log2(...))` and `SATFINITE` *is* the
clamp.

## The pipeline, and why BF16 sits in the middle

```
DeepSeek-V4-Flash-0731        release: experts packed MXFP4, rest block-scaled FP8
  │  tools/fp8_cast_bf16.py       <- MR !1 changes this step
  ▼
DeepSeek-V4-Flash-0731-bf16   567 GB, uniformly BF16
  ├─ tools/convert_hf_to_mxfp8.py ──► ...-MXFP8  296 GB  → phase 1 rollout
  └─ tools/convert_hf_to_torch_dist.py ──► ..._torch_dist → trainer, both phases
```

BF16 is not an artifact anyone wants; it is where the decode lands. The trainer
cannot start from 4-bit weights — MXFP8 training keeps BF16 master weights and
quantizes per GEMM — and `convert_hf_to_torch_dist.py` reads ordinary HF BF16
tensors, so the decode has to happen once regardless. Two consumers need it, so
it is written once and shared. It can be deleted after both downstream artifacts
exist. Phase 2's rollout does not use this chain at all; it serves the release
directly.

## Things whose names mislead

- **`--ref-load` is miles', not Megatron's.** Its stated job is the reference
  policy, but with no `--load` it also supplies the initial training weights,
  which is the only reason this experiment passes it.
- **The recipe runs GRPO, not PPO** — `--advantage-estimator grpo`, no value
  network, advantages normalized within each prompt's group of 8 samples.
- **The KL penalty is off.** `--kl-loss-coef 0.00`, and `losses.py:305` gates on
  it, so the reference model contributes nothing to the loss here.
- **`train_rollout_kl` is not that KL.** It measures the trainer and the rollout
  engine disagreeing about the same current weights — train/inference
  consistency, not policy drift. It is the experiment's whole observable, and it
  is unaffected by the reference model.

## Still unmeasured

`SKIP_SAVING=1` throughout. No run here has exercised checkpoint saving, so none
of these numbers is comparable to a run that does. Removing it is one
`submit.sh phase1 SKIP_SAVING=0` away and nobody has spent it.

The intermittent phase 1 hang is not root-caused either. It presents as one
rollout taking two to four times as long as its predecessors until the 300 s
watchdog fires, and the watchdog's own py-spy dump puts the scheduler in a
native CUDA wait inside the first prefill after a weight update. It did not
appear in 431471.
