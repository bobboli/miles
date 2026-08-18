# Handoff: 0731 MXFP8 RL, updated 2026-08-18

Written for whoever picks this up next. It says where the work stands, what is
open, and what it costs to be wrong about each piece. It does not repeat
[STATUS.md](STATUS.md) — that is the evidence file, and every claim here points
into it. Read [INDEX.md](../INDEX.md) first if you have not.

## Direct-HF P1/P2 training smoke passes on OCI

The no-offline-conversion bring-up is complete for one no-R3 training smoke in
each phase. Both jobs used `init_model_source=hf`,
`rollout_weight_source=trainer`, `load_format=dummy` in SGLang, 8 OCI nodes / 32
GPUs and one real optimizer step. Neither trainer loaded a BF16 HF conversion or
a `torch_dist` seed. P1 still uses a 6.2 MB metadata-only MXFP8 schema to tell
SGLang how to allocate tensors; it contains no weights or weight index.

| Phase | Job | Covered path | Result |
|---|---:|---|---|
| P1: MXFP8 rollout | `501870` | initial online update, real generation, log-prob, actor train, post-train update | `COMPLETED 0:0`, 59:25 |
| P2: FP8 + packed-MXFP4 experts | `501657` | same, including MXFP4 restore/repack and memory-saver lifecycle | `COMPLETED 0:0`, 54:57 |

All 32 trainer ranks completed both update transactions. P1's initial and
post-train updates took about 429 s and 226 s; P2 took 389 s and 164 s. The
generated batches were healthy enough for a smoke: P1 reported
`raw_reward=0.546875`, `truncated_ratio=0.558594`; P2 reported `0.710938` and
`0.371094`. In particular, P2 no longer reproduces the pre-fix zero-reward,
near-100%-truncation failure.

Both jobs explicitly had `enable_r3=False` and
`use_rollout_routing_replay=False`. SGLang MR
[!1](https://gitlab-master.nvidia.com/lbo/sglang/-/merge_requests/1) now contains
only the TRT-LLM MXFP4 hot-reload/memory-saver fixes. R3 `HashTopK` capture is a
separate MR [!2](https://gitlab-master.nvidia.com/lbo/sglang/-/merge_requests/2)
and was not applied to these jobs.

Do not overstate the result. Each job performed one optimizer step and two
weight reloads, but did not generate again after the second reload. Saving,
resume, two-or-more optimizer steps, R3 and direct-HF/offline-seed numerical
parity remain unverified. `train_rollout_kl` was still 0.1267 for P1 and 0.1310
for P2. At shutdown, Ray actors also emitted CUDA IPC unlink and W&B broken-pipe
atexit stacks after the work had succeeded; both Ray submissions reported
success and both Slurm jobs exited `0:0`, so this is a teardown issue rather
than a runtime failure.

Validated source revisions:

- Miles `dsv4-direct-hf-rollout`: `5cb48105e`;
- Megatron-Bridge `miles-dsv4-direct-hf`: `ca1a03de`;
- SGLang TRT-only MR !1: `605830cd5c`;
- separate, unused R3 MR !2: `8007bb9d31`.

## Root cause confirmed on local B300

The packed-MXFP4 failure is caused by one unregistered runtime tensor in
SGLang's FlashInfer TRT-LLM MoE method, not by the transferred weights.
`Mxfp4FlashinferTrtllmMoEMethod.create_moe_runner` creates
`_gemm1_clamp_limit_tensor` as a plain CUDA tensor containing the DeepSeek-V4
SwiGLU limit (`10.0` for every local expert). Model construction occurs inside
TorchMemorySaver's `weights` region. With CPU backup disabled,
`pause("weights")` discards the tensor contents and `resume("weights")` maps the
same virtual storage back as zeros. The tensor is neither a parameter nor a
buffer, so SGLang's static-state backup, the Miles update and the weight checker
all miss it. Every TRT-LLM MXFP4 MoE call nevertheless passes it as
`gemm1_clamp_limit`.

The complete production-order failure and fix reproduce on this node without a
trainer:

| Container | Change | Checker | First generation |
|---|---|---|---|
| `dsv4-miles-initial-sync-memsaver-hchead` | full 67,569-tensor sync, memory saver on | `Success` | `4/4` hit 1024; corrupt repetition |
| `dsv4-miles-initial-sync-memsaver-nograph` | same, CUDA graph disabled | `Success` | `4/4` hit 1024; corrupt repetition |
| `dsv4-miles-initial-sync-memsaver-clampfix` | register the clamp tensor as a non-persistent buffer | `Success`, including all 43 clamp buffers | `0/4` truncated; token counts 26/45/21/124; all `finish=stop` |

A standalone TorchMemorySaver check in the same image changed a tensor allocated
in the `weights` region from `[10.0]` before pause to `[0.0]` after
pause/resume. The real 64-expert TRT-LLM kernel probe then changed from digest
`cbbad8e8e1fe2710` at clamp 10 to `07854d2fef297a06` at clamp 0, with relative
L2 exactly 1.0; restoring 10 recovered the original output bit for bit. This,
the no-graph reproduction and the end-to-end fixed run close the causal chain.

The minimal fix is in
[lbo/sglang!1](https://gitlab-master.nvidia.com/lbo/sglang/-/merge_requests/1):
register the existing tensor on the expert layer with `persistent=False`. That
makes the existing memory-saver static-state export/import preserve it and also
makes the weight checker cover it. No Miles, quantization or kernel API change
is required.

## Direct-HF P1/P2 initial sync now passes locally

The current workspace also removes the offline conversion requirement for a
fresh P1/P2 run. `scripts/run_deepseek_v4.py` now separates the trainer seed
(`init_model_source`) from the rollout allocation layout
(`rollout_weight_source`). With `hf + trainer`, Megatron-Bridge directly imports
the official hybrid checkpoint into BF16 trainer parameters, SGLang starts with
`load_format=dummy`, and the initial trainer update supplies every rollout
tensor.

P1 uses a 6.2 MB metadata-only MXFP8 schema generated from safetensors headers;
it contains config/tokenizer files but no weight index or tensor payload. P2
uses the official config to allocate FP8 non-routed weights and packed MXFP4
experts, but also skips checkpoint tensor loading. Both were exercised on this
node with 8×B300 and two TP4/EP4 colocated engines, zero optimizer steps:

| Container | Result |
|---|---|
| `dsv4-direct-hf-p1-initial-sync-v2` | all Bridge tasks and buckets loaded; both `end_weight_update` and `continue_generation` returned 200; 8 trainer ranks `ok=true`; exit 0 |
| `dsv4-direct-hf-p2-initial-sync-v2` | same with `flashinfer_mxfp4`; update completed in 615--617 s, including about 358 s cold MHC/TileLang compile; exit 0 |

The first P1 attempt exposed a real bucket bug: Bridge quantization emits both
weight and scale from one Megatron parameter, but the stream retained only one
derived tensor per atomic-group slot. The iterator now keeps every consecutive
derived tensor together before combining DSV4 cross-parameter groups. P2 then
completed all buckets and the final MXFP4 repack.

This local result is now supplemented by the one-step OCI P1/P2 smoke above.
The full design and remaining validation boundary are in
[`work/2026-08-17/dsv4-mxfp4-checkpoint-mxfp8-training-flow.md`](../../work/2026-08-17/dsv4-mxfp4-checkpoint-mxfp8-training-flow.md).

## 2026-08-17 local B300 update

The packed-MXFP4/flattened-CUDA-IPC cell now has a strict full-model control.
The official release at
`/home/scratch.lbo_other/dsv4_flash_rl/models/DeepSeek-V4-Flash-0731` was served
on one local B300 node with TP4/EP4 and `flashinfer_mxfp4`. GPU 0-3 hosted the
engine and their rank-local CUDA payloads. The early control used one begin/end
transaction for all 48 checkpoint shards and 1 GiB flattened buckets, but it
did **not** exercise the rollout's memory release: its engine had
`enable_memory_saver=False`, so the release/resume API calls were no-ops. That
omission is what hid the root cause until the production flag was reproduced.

The production failure is an **initial-sync** failure, not a post-optimizer
failure. `train.py` calls `actor_model.update_weights()` before the first
rollout and before any optimizer step. Job 440119 snapshots the freshly loaded
rollout model, poisons it, performs that initial trainer-to-rollout sync, and
successfully compares the finalized model before the first broken generation.

The first apparent full-model reproduction was invalid. The release stores
`attn.wo_a.weight` as block-FP8 `[8192, 4096]`, while this run deliberately sets
`SGLANG_OPT_FP8_WO_A_GEMM=0`, so each rollout rank holds a BF16
`[2048, 4096]` shard. Replaying the checkpoint FP8 values directly cast numbers
as large as 448 into that BF16 parameter instead of dequantizing them. The
weight checker reported `max_abs_err=447.890625` on every `wo_a`, and generation
went from `0/4` to `4/4` truncated. That was a harness format error, not an
online-update failure.

The corrected `miles_full` replay does what the Miles source path does at the
initial trainer weights: it multiplies each `wo_a` FP8 block by its 128x128
UE8M0 scale, sends the resulting BF16 weight, and omits the scale because the
runtime BF16 parameter has none. All other release tensors are converted to the
names and scale layouts emitted by Miles. The early replay sent 67,566 tensors
because it also skipped the three `hc_head_*` tensors; the production-faithful
control sends 67,569. The early replay exited zero:

| Measurement | Result |
|---|---|
| valid full-model update | weight checker `Success`; `0/4` truncated before and after; all four texts identical |
| `dsv4-miles-initial-sync-full-strict`, full model before first generation | 67,566 tensors; `pause/continue`; checker `Success` before KV/graph resume; first generation `0/4` truncated; exit 0 |
| BF16 expert x1.05, then Miles MXFP4 requantization | 66,048 expert tensors updated; `0/4` truncated after update; restoring the release gives `0/4`, with all four baseline texts exact |
| same x1.05 update, four independent producers, batch 256 | checker snapshot/restore `Success`; baseline, changed and restored runs each have the expected `64/256` deliberately capped requests; every completed answer restores |
| experts + nonexpert weights, excluding `wo_a` | `0/4` truncated before and after; answers remain correct |
| checkpoint FP8 copied into BF16 `wo_a` | checker fails with `max_abs_err=447.890625`; `0/4` to `4/4` truncated (invalid control) |
| identity packed-expert update | `0/4` truncated before and after; all four texts identical |
| changed-layout unit contract | a second, different weight/scale payload matches a fresh load byte for byte while kernel-layout addresses stay fixed |
| negate both E2M1 sign bits | `0/4` to `4/4` truncated; restoring the release returns `0/4` and the four correct answers |
| increment every UE8M0 scale exponent | coherent baseline to immediate garbage; restoring the release returns `0/4` and text identical to baseline |

Partial weight-only or scale-only selectors are useful behavioral bisections,
but they are not strict identity controls: block-FP8 weights and scales form one
quantized value. The full corrected replay is the one whose post-update checker
passes.

This exonerates the initial-value Miles-format update itself, packed MXFP4
restore/repack, all visible expert and nonexpert parameters, 1 GiB flattened
CUDA IPC and the TP4/EP4 topology on local B300. It did not exonerate the real
memory cycle because memory saver was disabled. It also
exonerates ordinary value changes as the missing ingredient. A second TP4/EP4
run dequantized every release expert to BF16, multiplied it by 1.05, and called
the current Miles `mxfp4_quantize` before the same update lifecycle. On a real
expert tensor, 44.86% of packed bytes and 45.95% of scale bytes changed while
the represented value moved by relative L2 0.02066. All 66,048 expert tensors
were accepted; generation remained coherent and correct with `0/4` truncation.
Replaying the release again restored `0/4` and all four baseline texts exactly.

The rollout-sized control also passes. Four persistent spawned processes, one
per TP rank on CUDA 0-3, independently build and retain their flattened IPC
buckets. The engine captures and uses the batch-256 decode CUDA graph, receives
the same 66,048 changed expert tensors in one 48-shard transaction, and runs 256
requests before the update, after it and after restore. Each pass reports
`64/256` at the 64-token test cap: those are the 64 copies of the deliberately
open-ended `the sky is` prompt, not corruption. The other 192 requests complete,
and all 192 final answers match after restore. Some hidden-reasoning text is not
byte-identical at this batch size, but the strict SGLang tensor checker compares
the restored kernel-layout model against its pre-update snapshot and reports
`Success`. The container
`dsv4-miles-requantize-highbatch-strict-roundtrip` exited zero with
`ROUND TRIP RESTORED`.

The initial-sync ordering itself is not the missing condition. The earlier
`dsv4-miles-initial-sync-full-strict` control followed that ordering and passed,
but its release/resume calls were no-ops. Repeating it with
`enable_memory_saver=True` and all 67,569 tensors is the local reproducer listed
above: the visible weights compare `Success`, then first generation is broken.

One real fused-conversion check closes another part of the producer path. The
release's layer-0 expert-0 `w1` and `w3` were dequantized to BF16, concatenated
into the Megatron `linear_fc1` layout, then passed through Miles'
`convert_to_hf` and MXFP4 processor. Both emitted gate/up weights and both
UE8M0 scales reproduced the release bytes exactly. This covers the fused-FC1
split, names and quantization entry point; it does not cover the distributed
PP/EP/TP gather.

The quantizer check now covers the complete release rather than one sample.
Across 46 shards containing packed experts, all 35,328 expert weight tensors
(33,024 main-model tensors plus 2,304 speculative tensors) survive release
MXFP4 -> BF16 -> the current Miles `mxfp4_quantize` byte for byte, for both
packed payload and UE8M0 scale. Eight local GPUs ran the read-only check and the
container exited zero.

A fixed-input kernel check reaches below generation and the tensor checker.
The production EP-rank complement of 64 real layer-0 experts was put through
the exact MXFP8 activation quantization, top-8 packing, TRT-LLM shuffle and
FlashInfer `trtllm_fp4_block_scale_routed_moe` call. The 64 fixed tokens route
across all 64 local experts. A repeated fresh call, an identity restore/repack,
and a restore after a changed update all reproduced the BF16 output bit for bit
(`max_abs=0`, SHA-256 prefix `cbbad8e8e1fe2710`). Negating both packed E2M1
sign bits changed the output by relative L2 1.53915, proving the call read the
updated payload. Setting only the SwiGLU clamp tensor from 10 to zero changed the
same output by relative L2 1.0; restoring 10 was bitwise exact. The container
`dsv4-mxfp4-frozen-moe-output` exited zero.

Source inspection also makes a stale FlashInfer tensor pointer unlikely. The
Python cache retains the loaded op module and tactic IDs, not weights or tensor
addresses. Each raw `trtllm_fp4_block_scale_moe` call creates its launcher map,
runner and workspaces again, and `prepare_moe` reads the current weight, scale
and buffer `data_ptr()` values. SGLang's three output-scale buffers are
re-registered during post-load processing, but the current buffer addresses are
passed into that per-call setup.

The previous local/OCI mismatch is now closed inside the one-engine lifecycle.
The local reproducer uses the same image index
`sha256:08be00658cd24eaa364ca4ad0b1a3911dfbe4adc04fd0c148e4241402fb40812`,
TP4/EP4, 1 GiB rank-local CUDA-IPC buckets, production initial-sync ordering and
real TorchMemorySaver behavior. Neither eight-node Megatron orchestration nor
the arm64 GB300 platform is required to trigger the failure.

The experiment patch version is not the missing difference. The local full
replay's patched MXFP4 source is byte-identical to the current patch, whose last
functional change was the in-place address fix already exercised by failing job
454574. The CUDA-IPC clone/synchronize patch has not changed since before the
bad jobs, and inspection of the retained local container confirms it was active
there too.

Capturing a real trainer payload is no longer needed to locate the failure. The
next cluster measurement is a verification run with the SGLang fix, not another
payload or kernel bisection.

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

The no-R3 direct-HF P1/P2 smoke and the packed-MXFP4 operational verification
are complete. The old P2 corruption (`raw_reward=0`,
`truncated_ratio=0.99–1.00`) does not reproduce with the SGLang fix. What remains
is broader correctness and lifecycle coverage, not another initial-sync
bisection:

- run at least two optimizer steps and generate after each post-train reload;
- investigate the still-high one-step `train_rollout_kl` before claiming parity;
- exercise save/resume and a reusable model-only `torch_dist` seed;
- validate R3 separately after MR !2 review;
- clean up the CUDA IPC/W&B atexit warnings without changing the now-passing
  runtime path.

### The next measurement

Extend the passing jobs to `num_rollout>=2`, so a second optimizer update and
generation after the second post-train reload are observed. Keep R3 off for that
control. Do not repeat payload capture, graph, bucket or weight-layout
bisections unless the longer run contradicts the passing one-step result.

Two dead ends, so they are not retried:

- **Swapping the MXFP4 kernel** does not isolate weights from kernel here.
  `marlin` needs SM90 or SM120 and GB200 is SM100 — a 45-minute run learned what
  a minute of reading the image would have said. `humming` has no
  `restore_load_layout`, so it fails a second load for the reason this work
  already fixed elsewhere.
- **Partial checkpoint replays are not identity controls.** Closing begin/end
  before all shards arrive postprocesses a partial model; splitting block-FP8
  weights from their scales changes the represented values; copying release
  FP8 `wo_a` into runtime BF16 is a dtype conversion, not dequantization.

## Merge requests open for review

All on GitLab, none visible outside NVIDIA. Review order is the table order —
the first two are independent of the MXFP4 work and of each other.

| MR | Contents |
|---|---|
| [lbo/miles!1](https://gitlab-master.nvidia.com/lbo/miles/-/merge_requests/1) | MXFP4 → BF16 decode in `tools/fp8_cast_bf16.py`, plus `miles/utils/mxfp4.py` and 12 CPU-only tests |
| [lbo/miles!2](https://gitlab-master.nvidia.com/lbo/miles/-/merge_requests/2) | Weight-update IPC lifetime and the post-update checksum |
| [lbo/miles!3](https://gitlab-master.nvidia.com/lbo/miles/-/merge_requests/3) | Registering the release and configuring rollout from its actual layout |
| [lbo/miles!4](https://gitlab-master.nvidia.com/lbo/miles/-/merge_requests/4) | This directory: the run log, the tools, the retrospective |
| [lbo/sglang!1](https://gitlab-master.nvidia.com/lbo/sglang/-/merge_requests/1) | TRT-LLM MXFP4 kernel-layout hot reload plus the TorchMemorySaver clamp-buffer fix |
| [lbo/sglang!2](https://gitlab-master.nvidia.com/lbo/sglang/-/merge_requests/2) | R3 `HashTopK` routed-expert capture, deliberately separate from !1 |

The sglang MR targets `sglang-miles`, not `main`. The local release image
reports SGLang build commit `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`;
the Miles begin/end API lives on `sglang-miles`, not upstream main. The clamp
fix itself needs no Megatron change; the direct-HF path uses the separate
Megatron-Bridge revision listed above.

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
