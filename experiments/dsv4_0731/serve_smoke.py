"""Generate a few answers from the release checkpoint and show them.

The question is coarse and does not need a benchmark: a model that answers
nothing and never emits a stop token looks nothing like one that works. Prompts
have short, unambiguous answers so the transcript can be read directly, and the
finish reason is printed because running to the length limit is the symptom
phase 2 shows.

Configured by the environment its sbatch exports: ``MODEL``, ``FP4_EXPERTS``
and ``MOE_BACKEND``.
"""

from __future__ import annotations

import os

import sglang

PROMPTS = [
    "What is 2 + 2? Reply with only the number.",
    "Name the capital of France. Reply with only the city name.",
    "Compute 17 * 3. Reply with only the number.",
    "Complete the sentence with one word: the sky is",
]
MAX_NEW_TOKENS = 128


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

    tokenizer = engine.tokenizer_manager.tokenizer
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in PROMPTS
    ]

    outputs = engine.generate(
        prompts,
        sampling_params={"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS},
    )

    truncated = 0
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
    print(f"{truncated}/{len(PROMPTS)} responses ran to the token limit")
    engine.shutdown()


if __name__ == "__main__":
    main()
