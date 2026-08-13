"""Generate, feed the model its own weights back, and generate again.

Phase 2 answers nothing after a weight update, on weights a post-update
comparison certifies as correct, through a serving path that works untouched.
Isolating what the update breaks has so far cost an eight-node run per attempt.

An update that hands the model exactly the weights it already holds should be
invisible. If generation degrades across it, the whole failure reproduces on one
node with no trainer, no quantizer and no miles code in the path — and the cycle
becomes twenty-five minutes on four GPUs.

Configured by the environment its sbatch exports: ``MODEL``, ``FP4_EXPERTS``,
``MOE_BACKEND``, ``UPDATE_SELECTOR``, ``MEMORY_CYCLE``, ``TRANSPORT``, ``BUCKET_BYTES`` and ``MEM_FRACTION``.
"""

from __future__ import annotations

import glob
import os

import sglang
import torch
from safetensors import safe_open
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

from miles.utils.chat_template_utils import deepseek

PROMPTS = [
    "What is 2 + 2? Reply with only the number.",
    "Name the capital of France. Reply with only the city name.",
    "Compute 17 * 3. Reply with only the number.",
    "Complete the sentence with one word: the sky is",
]
MAX_NEW_TOKENS = 1024
BUCKET_BYTES = int(os.environ.get("BUCKET_BYTES", 64 << 20))


def render(tokenizer) -> list[str]:
    return [
        deepseek.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenizer, thinking_mode="thinking"
        )
        for prompt in PROMPTS
    ]


def report(engine, prompts: list[str], label: str) -> int:
    outputs = engine.generate(
        prompts, sampling_params={"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS}
    )
    truncated = 0
    print("=" * 72)
    print(label)
    for prompt, output in zip(PROMPTS, outputs, strict=True):
        meta = output.get("meta_info", {})
        finish = meta.get("finish_reason", {})
        reason = finish.get("type") if isinstance(finish, dict) else finish
        truncated += reason == "length"
        print("-" * 72)
        print(f"prompt: {prompt}")
        print(f"tokens: {meta.get('completion_tokens')}  finish: {reason}")
        print(f"output: {output['text']!r}")
    print("-" * 72)
    print(f"{label}: {truncated}/{len(PROMPTS)} ran to the token limit")
    return truncated


def batched(named, budget: int):
    """Split a shard into buckets of roughly *budget* bytes, as the rollout does.

    A whole shard at once needs its own copy on the device beside the engine's
    weights and pool, which does not fit. The rollout sizes its buckets by
    ``--update-weight-buffer-size`` for the same reason.
    """
    batch, size = [], 0
    for name, tensor in named:
        batch.append((name, tensor))
        size += tensor.numel() * tensor.element_size()
        if size >= budget:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def send(engine, named: list[tuple[str, torch.Tensor]]) -> None:
    """Hand a batch of tensors over the transport phase 2 uses, or the plain one.

    The rollout path does not pass tensors: it flattens each batch into one
    buffer, shares it over CUDA IPC, and lets every rank reconstruct views into
    it. That is a different set of lifetimes from a plain hand-off, so it is
    worth being able to exercise either.
    """
    if os.environ.get("TRANSPORT", "bucket") != "bucket":
        engine.update_weights_from_tensor(named, flush_cache=False)
        return

    bucket = FlattenedTensorBucket(
        named_tensors=[(name, tensor.cuda()) for name, tensor in named]
    )
    payload = {
        "flattened_tensor": bucket.get_flattened_tensor(),
        "metadata": bucket.get_metadata(),
    }
    serialized = MultiprocessingSerializer.serialize(payload, output_str=True)
    engine.update_weights_from_tensor(
        [serialized] * engine.server_args.tp_size,
        load_format="flattened_bucket",
        flush_cache=False,
    )
    # The exporting storage has to outlive every importer's use of the handle,
    # and this process shares its GPU with a worker, so give the block back.
    torch.cuda.synchronize()
    del payload, bucket
    torch.cuda.empty_cache()


def identity_update(engine, model: str, selector: str) -> None:
    """Hand the model back the tensors it was loaded from, shard by shard.

    Only the parameters *selector* names are sent. The attention scales cannot
    take part: the first load reshapes them, so their own checkpoint bytes no
    longer fit the parameter they came from, and a second load asserts. The
    routed experts are what phase 2 changes and what this needs to exercise.
    """
    shards = sorted(glob.glob(os.path.join(model, "*.safetensors")))
    transport = os.environ.get("TRANSPORT", "bucket")
    print(f"identity update over {len(shards)} shards, selector={selector!r}, transport={transport}, bucket={BUCKET_BYTES >> 20} MiB")
    # Imported buffers accumulate until the window closes, and this process
    # shares its GPU with a worker, so the window closes every few shards.
    shards_per_window = int(os.environ.get("SHARDS_PER_WINDOW", "4"))
    sent = 0
    engine.begin_weight_update()
    for index, shard in enumerate(shards):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            named = [
                (name, handle.get_tensor(name))
                for name in handle.keys()
                if selector in name
            ]
        for batch in batched(named, BUCKET_BYTES):
            send(engine, batch)
            sent += len(batch)
        del named
        closing = (index + 1) % shards_per_window == 0 or index == len(shards) - 1
        if closing:
            engine.end_weight_update()
            print(f"  shard {index + 1}/{len(shards)}, {sent} tensors sent, window closed")
            if index != len(shards) - 1:
                engine.begin_weight_update()
    torch.cuda.synchronize()
    print(f"identity update complete, {sent} tensors")


def main() -> None:
    model = os.environ["MODEL"]
    print(f"model={model}")
    print(f"SGLANG_DSV4_FP4_EXPERTS={os.environ.get('SGLANG_DSV4_FP4_EXPERTS')}")
    print(f"moe_runner_backend={os.environ['MOE_BACKEND']}")

    engine = sglang.Engine(
        model_path=model,
        tp_size=4,
        ep_size=4,
        trust_remote_code=True,
        attention_backend="dsv4",
        moe_runner_backend=os.environ["MOE_BACKEND"],
        # The bucket is built in this process while a worker holds the same GPU,
        # so the engine has to leave room the rollout would not need to.
        mem_fraction_static=float(os.environ.get("MEM_FRACTION", "0.45")),
        kv_cache_dtype="fp8_e4m3",
        page_size=256,
        skip_server_warmup=True,
        log_level="info",
    )

    prompts = render(engine.tokenizer_manager.tokenizer)
    before = report(engine, prompts, "before update")

    # The rollout engine gives its memory back while training runs and takes it
    # again around the update, in this order. Weights return before the update
    # so it has somewhere to land; the KV cache returns after.
    cycle = os.environ.get("MEMORY_CYCLE", "1") == "1"
    if cycle:
        print("release_memory_occupation()")
        engine.release_memory_occupation()
        print(f"resume_memory_occupation(tags=[{GPU_MEMORY_TYPE_WEIGHTS}])")
        engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    identity_update(engine, model, os.environ.get("UPDATE_SELECTOR", ".ffn.experts."))

    if cycle:
        print(f"resume_memory_occupation(tags=[{GPU_MEMORY_TYPE_KV_CACHE}])")
        engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])

    after = report(engine, prompts, "after update")

    print("=" * 72)
    print(f"truncated before={before}/{len(PROMPTS)} after={after}/{len(PROMPTS)}")
    print("REPRODUCED" if after > before else "not reproduced")
    engine.shutdown()


if __name__ == "__main__":
    main()
