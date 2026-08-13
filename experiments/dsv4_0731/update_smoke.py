"""Generate, feed the model its own weights back, and generate again.

Phase 2 answers nothing after a weight update, on weights a post-update
comparison certifies as correct, through a serving path that works untouched.
Isolating what the update breaks has so far cost an eight-node run per attempt.

An update that hands the model exactly the weights it already holds should be
invisible. If generation degrades across it, the whole failure reproduces on one
node with no trainer, no quantizer and no miles code in the path — and the cycle
becomes twenty-five minutes on four GPUs.

Configured by the environment its sbatch exports: ``MODEL``, ``FP4_EXPERTS``,
``MOE_BACKEND`` and ``UPDATE_SELECTOR``.
"""

from __future__ import annotations

import glob
import os

import sglang
import torch
from safetensors import safe_open

from miles.utils.chat_template_utils import deepseek

PROMPTS = [
    "What is 2 + 2? Reply with only the number.",
    "Name the capital of France. Reply with only the city name.",
    "Compute 17 * 3. Reply with only the number.",
    "Complete the sentence with one word: the sky is",
]
MAX_NEW_TOKENS = 1024


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


def identity_update(engine, model: str, selector: str) -> None:
    """Hand the model back the tensors it was loaded from, shard by shard.

    Only the parameters *selector* names are sent. The attention scales cannot
    take part: the first load reshapes them, so their own checkpoint bytes no
    longer fit the parameter they came from, and a second load asserts. The
    routed experts are what phase 2 changes and what this needs to exercise.
    """
    shards = sorted(glob.glob(os.path.join(model, "*.safetensors")))
    print(f"identity update over {len(shards)} shards, selector={selector!r}")
    engine.begin_weight_update()
    sent = 0
    for index, shard in enumerate(shards):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            named = [
                (name, handle.get_tensor(name))
                for name in handle.keys()
                if selector in name
            ]
        if named:
            engine.update_weights_from_tensor(named, flush_cache=False)
            sent += len(named)
        if index % 8 == 0 or index == len(shards) - 1:
            print(f"  shard {index + 1}/{len(shards)}, {sent} tensors sent")
        del named
    engine.end_weight_update()
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
        mem_fraction_static=0.6,
        kv_cache_dtype="fp8_e4m3",
        page_size=256,
        skip_server_warmup=True,
        log_level="info",
    )

    prompts = render(engine.tokenizer_manager.tokenizer)
    before = report(engine, prompts, "before update")
    identity_update(engine, model, os.environ.get("UPDATE_SELECTOR", ".ffn.experts."))
    after = report(engine, prompts, "after update")

    print("=" * 72)
    print(f"truncated before={before}/{len(PROMPTS)} after={after}/{len(PROMPTS)}")
    print("REPRODUCED" if after > before else "not reproduced")
    engine.shutdown()


if __name__ == "__main__":
    main()
