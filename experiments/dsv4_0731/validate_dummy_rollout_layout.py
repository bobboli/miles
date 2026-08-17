"""Build a DeepSeek V4 rollout layout without loading checkpoint weights."""

from __future__ import annotations

import argparse
import os

import sglang


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--mem-fraction", type=float, default=0.5)
    parser.add_argument("--moe-runner-backend", required=True)
    parser.add_argument("--fp8-gemm-backend", default="flashinfer_cutlass")
    parser.add_argument("--enable-memory-saver", action="store_true")
    parser.add_argument("--disable-flashinfer-autotune", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"model={args.model}")
    print(f"SGLANG_DSV4_FP4_EXPERTS={os.environ.get('SGLANG_DSV4_FP4_EXPERTS')}")
    engine = sglang.Engine(
        model_path=args.model,
        load_format="dummy",
        tp_size=args.tp_size,
        ep_size=args.tp_size,
        trust_remote_code=True,
        attention_backend="dsv4",
        fp8_gemm_runner_backend=args.fp8_gemm_backend,
        moe_runner_backend=args.moe_runner_backend,
        mem_fraction_static=args.mem_fraction,
        kv_cache_dtype="fp8_e4m3",
        page_size=256,
        enable_memory_saver=args.enable_memory_saver,
        disable_flashinfer_autotune=args.disable_flashinfer_autotune,
        disable_cuda_graph=True,
        skip_server_warmup=True,
        log_level="info",
    )
    print(f"DUMMY_ROLLOUT_LAYOUT_READY model={engine.server_args.model_path} load_format={engine.server_args.load_format} moe={engine.server_args.moe_runner_backend}")
    engine.shutdown()


if __name__ == "__main__":
    main()
