# Getting MXFP8 RL running on 0731: what it took, and why it was hard

Written after phase 1 completed four steps (job 431471, 2026-08-12). Phase 2 has
no measurement yet; that gap is listed at the end rather than smoothed over.

## What we changed

Ten commits, about 1,600 lines. Almost none of it is MXFP8 support as such — the
recipe already accepted `--train-mxfp8` and `--rollout-mxfp8`. What had to be
built was a converter for a checkpoint layout nothing had converted before, and
repairs to an online weight-update path that had never been exercised in these
formats.

| Area | Change |
|---|---|
| Offline conversion | `tools/fp8_cast_bf16.py` picks its branch from the scale extent, so packed MXFP4 and block-scaled FP8 both decode; scales resolve in the checkpoint's own namespace; a weight with no resolvable scale raises. `miles/utils/mxfp4.py` is the inverse. |
| Online update path | `processors/quantizer_mxfp4.py` and its dispatch; `--rollout-fp4-experts`; post-update checksums behind `--check-weight-update-equal`. |
| Recipe wiring | `DeepSeek-V4-Flash-0731` registered; MoE runner backend and `SGLANG_DSV4_FP4_EXPERTS` derived from the rollout checkpoint's layout; `PRECISION` gained `mxfp8` and `mxfp8-fp4r`; the launcher's patch list became generic. |
| SGLang | Three patches, none upstream: IPC clone plus collect after each update, MXFP4 kernel-layout hot reload, memory reporting. |
| Validation | `probe_mxfp4.py`, `validate_mxfp4_dequant.py`, `make_mini_checkpoint.py`, `assert_fully_dequantized.py`, and two unit-test files. |

## Why it was hard

**Nothing fails until the second load.** Step 0 serves weights that came through
`process_weights_after_loading`. Every defect found in this bring-up lives in the
*next* load — the first online update. Reaching that point costs about 25 minutes
of image import, checkpoint load and rollout. The first opportunity for a bug to
appear is also the most expensive moment in the run.

**Silent corruption was the default, not the exception.** Three separate paths
would have produced wrong numbers rather than an error:

- the converter copied a quantized payload through when it could not resolve a
  scale — this actually fired, on 2 tensors out of 36,600 in a 567 GB output;
- the MXFP4 kernel layout differs from the load layout in dtype, so refilling it
  casts instead of failing: a scale byte of 130 comes back as 112;
- MXFP4 scales are staged as floats, so handing them over as `uint8` collapsed
  the exponents 120, 127, 130 and 140 onto a single byte.

Only the first announced itself, and only because we added an artifact check.

**The instruments lied.** `torch.cuda.memory_allocated` reported 164 GiB while
the driver reported 83 GiB in use, because the memory saver releases pages and
keeps the bookkeeping. Anyone debugging memory from allocator statistics — which
is the obvious thing to do — gets a false picture. Separately, the scheduler
watchdog says "no progress for 300 s", which reads as a deadlock; the runs were
in fact suffering a throughput collapse, one rollout taking two to four times as
long as the ones before it.

**Reading the code produced three wrong root causes.** That the MoE shuffle was
not re-applied after an update; that finalization raced with resuming
generation; that memory accumulated across updates. The first was refuted by
post-update checksums, the second by implementing it and watching the run get
worse, the third by the plateau in this run's memory trace. Each cost at least
one eight-node run. In a codebase this size, a hypothesis derived from reading
is a candidate for measurement, not a conclusion.

**The configuration is unmanaged.** The launcher's defaults are not the
configuration any run here uses. Submitting with them moved six settings at once
and cost a run to an assertion that had nothing to do with the work. Nothing in
the repository records which environment a phase needs.

**The SGLang changes live outside version control of the thing that runs them.**
Three patches re-applied to the image on every job, verified only by whether they
apply. An image refresh breaks them, and until this week the verification step
was hardcoded to one patch's contents, so any other patch failed the job on a
grep.

## What to change

Ordered by what would have saved the most time here.

1. **Make the second load a test rather than a run.** A single-node job that
   loads the model, performs one synthetic weight update, and compares the
   parameters byte for byte would have caught all three MXFP4 defects in
   minutes. We ended up building that comparison by hand, once per defect,
   after each had already cost a full run.
2. **Make numeric hand-over failures loud.** The converter now raises; the
   update path should too. `--check-weight-update-equal` was switched off for
   phase 2 because it does not model kernel layouts — that is the wrong
   direction. It should understand them.
3. **Report `mem_get_info` wherever allocator statistics are reported**, at
   least while the memory saver is on. The two numbers differing by 80 GiB with
   no warning is a trap.
4. **Make the watchdog report throughput.** "No progress for 300 s" should carry
   what the rollout's token rate had been doing for the preceding minutes. The
   distinction between a stall and a slowdown was only recoverable by comparing
   `perf/rollout_time` across jobs afterwards.
5. **Pin each phase's environment in the repository** and have the launcher
   source it, so no run silently takes defaults.
6. **Upstream the three SGLang patches, or pin the image digest.** Today the
   experiment is one image refresh from failing in a way that looks unrelated.
7. **Reproduce the throughput collapse on demand.** It is the one open failure,
   it did not appear in 431471, and nothing yet explains why 427089's third
   rollout took 1001 s when its first two took 235 s.

## Still open

- Phase 2 has produced no mismatch number. Its weight hand-over works; its
  rollout has never finished a step.
- The throughput collapse is not root-caused, and the three causes proposed so
  far were all wrong.
- `SKIP_SAVING` has been `1` throughout. No run here has exercised checkpoint
  saving, and the numbers above are not comparable to any run that does.
