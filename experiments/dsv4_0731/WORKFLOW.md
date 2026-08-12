# Development workflow for DeepSeek-V4 RL on AGA

How code gets written, validated, and run for this experiment. Written after the
0731 MXFP8 bring-up, and shaped by what went wrong there.

## Three tiers

Work belongs at the cheapest tier that can answer the question. Almost everything
that failed on the cluster in this experiment could have been caught earlier.

| Tier | Where | Turnaround | What belongs here |
|---|---|---|---|
| T0 | this host, `radixark/miles:latest` on a local B300 | seconds | unit tests, numeric validation against real tensors, end-to-end smoke tests on a sliced checkpoint |
| T1 | one cluster node under `--qos=short` | ~10 min plus queue | full-checkpoint conversion, anything needing the shared filesystem |
| T2 | eight cluster nodes | 25 min to the first weight sync, ~95 min for four steps | the RL run itself |

The local host runs the same image as the cluster and its B300 reports the same
compute capability as GB300, so MXFP4/MXFP8 kernel behaviour matches. Use it.

```bash
docker run --rm --gpus '"device=0"' --user $(id -u):$(id -g) \
  -e HOME=/tmp -e USER=$USER \
  -v $PWD:/workspace/miles -w /workspace/miles \
  radixark/miles:latest bash -lc '<command>'
```

`--user` keeps files written into the mount owned by you; without it the
container cannot write to the bind mount. `USER` must be set or
`getpass.getuser()` fails on the unmapped uid.

## What the image provides, and which code actually runs

`radixark/miles:latest`, digest
`sha256:08be00658cd24eaa364ca4ad0b1a3911dfbe4adc04fd0c148e4241402fb40812`.
It already contains a complete, installed stack — torch 2.11+cu130, Transformer
Engine 2.17, and editable installs of miles, SGLang and Megatron-core. Nothing
needs to be built or pip-installed to run a job.

That raises the question this layout keeps provoking: when a checkout is also
mounted, which copy executes? It depends on the package.

| Package | Installed in the image at | What actually runs |
|---|---|---|
| `miles`, `miles_plugins`, `scripts/`, `tools/` | editable, `/root/miles` | **your mounted checkout** |
| `sglang` | editable, `/sgl-workspace/sglang/python` | **the image's copy** |
| `megatron-core` | editable, `/root/Megatron-LM` | the image's copy |
| `mbridge`, `transformer_engine`, `torch` | site-packages | the image's copy |

The override is `PYTHONPATH=/workspace/miles`, which the launcher exports and
which precedes site-packages, so `import miles` resolves into the bind mount and
the image's own copy at `/root/miles` never loads. Confirm it rather than assume
it:

```bash
PYTHONPATH=/workspace/miles python -c \
  "import miles, sglang, os; print(os.path.dirname(miles.__file__), os.path.dirname(sglang.__file__))"
# /workspace/miles/miles  /sgl-workspace/sglang/python
```

Nothing mounts over SGLang, and `PYTHONPATH` cannot reach it, so SGLang changes
have to be applied to the image's copy inside each container:

```bash
patch --batch --forward -p1 -d /sgl-workspace/sglang < some.patch
```

`run_rl.sbatch` does this for every rank from the space-separated
`SGLANG_PATCH_FILE` list, and verifies the result compiles before starting Ray.
The effect is ephemeral: it is re-applied on every job and vanishes with the
container, which is what makes it safe to iterate on, and also why a patch that
stops applying after an image refresh fails the job rather than silently
reverting.

So, by kind of change:

- **miles, scripts, tools, plugins** — commit, push, update the cluster checkout;
  the next job picks it up. No rebuild.
- **SGLang** — write a `.patch` against `/sgl-workspace/sglang`, add it to
  `SGLANG_PATCH_FILE`. Check it applies to a copy of the file from *this* image
  before submitting; generating the diff from the image's own file guarantees it.
- **Megatron-core, TE, torch** — needs a new image. Out of scope for experiment
  work; treat their behaviour as fixed.

The image is pulled by tag, and pyxis re-imports all 40 GB at the start of every
job, costing several minutes before anything runs. Pin the digest for any result
you intend to keep, and cache a `.sqsh` on the shared filesystem to remove the
re-import.

## Validating a numeric change

The conversion work held up because each layer was checked against something
independent, in this order:

1. **Find the reference implementation before writing anything.** The MXFP4
   element order came out of SGLang's own `cast_e2m1fn_to_e4m3fn`, not from
   reading a spec. Guessing a nibble order costs a full conversion to discover.
2. **Compare against it bit-for-bit on real tensors.** One downloaded shard is
   enough; `validate_mxfp4_dequant.py` does this for both branches of the cast.
3. **Unit-test the parts a comparison cannot reach** — value tables, block
   exponents, layout discrimination — with no GPU and no SGLang import, so they
   run in CI.
4. **Smoke-test the whole entry point.** `make_mini_checkpoint.py` slices a real
   shard into a one-file checkpoint so `fp8_cast_bf16.py`'s `main` runs end to
   end. This is what caught the name-remap regression that unit tests could not:
   the remap silently degraded to identity and every tensor was copied through
   unconverted.
5. **Check the artifact, not just the code.** `assert_fully_dequantized.py`
   re-reads the emitted safetensors headers and fails if any quantized payload or
   orphaned scale survived. It caught two unconverted tensors in a 567 GB output
   that otherwise looked complete.

Rule of thumb: a conversion step that can fail silently needs an artifact check,
because "the job exited 0" says nothing about the bytes.

## Syncing code to the cluster

Use git against the internal GitLab mirror, not `rsync`.

```bash
# once
git remote add aga ssh://git@gitlab-master.nvidia.com:12051/lbo/miles.git
git remote add aga-checkout "oci-aga:$WORKSPACE/miles"

# each change
git add <files>            # only this experiment's files; the tree carries
git commit -s              # unrelated in-progress work that must not be committed
git push aga <branch>                                  # canonical history
git push aga-checkout <branch>                          # the cluster's copy
```

The cluster has no credential for GitLab and its sshd refuses agent forwarding,
so it cannot fetch from the mirror itself. Its checkout is therefore configured
as a deploy target, which updates its working tree from the push:

```bash
ssh oci-aga "cd $WORKSPACE/miles && git config receive.denyCurrentBranch updateInstead"
```

That only applies cleanly to a clean tree, which is the point: a job now always
runs a named commit, and `git log -1` on the cluster names it. GitLab holds the
canonical history; once the cluster has its own credential the second push
becomes a `git fetch aga` there.

Snapshot the cluster's tree onto a commit before the first switch. Its checkout
carried uncommitted work from an earlier experiment, and a checkout would have
discarded it.

Per-file `rsync` was the original approach and it failed twice in one session:
a job was submitted against a checkout missing a fix that had been made locally
minutes earlier, and another against a launcher that had never received the flag
it was meant to pass. Both cost a full queue-and-run cycle to discover.

Git also gives each submitted job a commit to name, which `rsync` cannot. Record
the SHA in the run notes so a result can be tied to the code that produced it.

If `rsync` is ever unavoidable, sync the whole tree rather than named files, and
never with `--delete`: the cluster checkout holds `wandb/`, `logs/` and `runs/`
directories belonging to jobs that are still running.

## Running on the cluster

- Account `coreai_devtech_all`, partition `batch`, QoS `short` — two hours, and
  `interactive` needs a reservation so it cannot be used for batch submission.
- `run_rl.sbatch` takes its configuration from the environment: `MODEL_NAME`,
  `PRECISION`, `NUM_ROLLOUT`, `SGLANG_MOE_RUNNER_BACKEND`,
  `SGLANG_FP8_GEMM_BACKEND`, `MILES_EXTRA_ARGS_APPEND`, and a space-separated
  `SGLANG_PATCH_FILE` list.
- Each job re-imports the 40 GB image through pyxis, which costs several minutes
  before anything starts. Caching a `.sqsh` on the shared filesystem would remove
  that.
- Monitor with `squeue`, `sacct` and the shared log. Do not attach with
  `srun --jobid`.

## Experiment discipline

The parts of this bring-up that went badly were process failures, not technical
ones:

- **Change one variable per run.** The only configuration that completed cleanly
  differed from the failing ones in two ways at once, and it took several runs to
  notice which one mattered.
- **Instrument before sweeping.** Adding a post-update weight check turned a
  thirty-minute hang into an immediate signal, and it was added only after five
  runs had already been spent. A run that yields a measurement is worth several
  that yield a pass/fail.
- **Do not infer a root cause from reading alone and then act on it.** Two causes
  were proposed from static reading of an unfamiliar codebase; both were refuted
  by the next run, and one of them was implemented first and made the failure
  worse. In a codebase this size, a reading-derived hypothesis is a candidate for
  measurement, not a conclusion.
- **Size the experiment to the question.** The train/rollout mismatch metrics are
  produced at step 0, so comparing rollout formats needs a single rollout, not
  four steps. Reach for the long run only when the question is about drift.
