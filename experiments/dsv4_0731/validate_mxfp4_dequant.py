"""Check ``tools/fp8_cast_bf16.mxfp4_dequant`` against the SGLang runtime decode.

SGLang loads DeepSeek-V4 MXFP4 routed experts through ``cast_e2m1fn_to_e4m3fn``,
which rescales each 128x128 tile onto a single UE8M0 exponent. Reconstructing the
weight from that output gives the values the inference engine actually sees, so
matching it proves the offline cast agrees with the runtime on element order,
value table, and block layout.
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from fp8_cast_bf16 import is_mxfp4_weight, mxfp4_dequant, to_float_scale, weight_dequant  # noqa: E402

FP8_BLOCK_SIZE = 128


def runtime_reference(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reconstruct the weight the SGLang FP4 expert path materializes."""
    from sglang.srt.layers.quantization.fp8 import cast_e2m1fn_to_e4m3fn

    weight_fp8, tile_scale = cast_e2m1fn_to_e4m3fn(packed, scale)
    expanded = (
        tile_scale.float()
        .repeat_interleave(FP8_BLOCK_SIZE, dim=0)
        .repeat_interleave(FP8_BLOCK_SIZE, dim=1)
    )
    return weight_fp8.float() * expanded


def compare(name: str, packed: torch.Tensor, scale: torch.Tensor) -> bool:
    ours = mxfp4_dequant(packed, scale).float()
    theirs = runtime_reference(packed, scale)

    exact = torch.equal(ours, theirs)
    mismatched = (ours != theirs).sum().item()
    denom = theirs.abs().clamp(min=torch.finfo(torch.float32).tiny)
    max_rel = ((ours - theirs).abs() / denom).max().item()
    print(
        f"  {name:<52s} exact={exact} mismatched={mismatched}/{ours.numel()} "
        f"max_rel={max_rel:.3e} absmax={theirs.abs().max().item():.6f}"
    )
    return exact


def compare_block_fp8(name: str, weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """Check the block-scaled FP8 branch against SGLang's block dequantizer."""
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

    ours = weight_dequant(weight, to_float_scale(scale)).float()
    theirs = block_quant_dequant(
        weight, to_float_scale(scale), [FP8_BLOCK_SIZE, FP8_BLOCK_SIZE], torch.float32
    ).float()

    exact = torch.equal(ours, theirs)
    max_rel = ((ours - theirs).abs() / theirs.abs().clamp(min=torch.finfo(torch.float32).tiny)).max().item()
    print(f"  {name:<52s} exact={exact} max_rel={max_rel:.3e} absmax={theirs.abs().max().item():.6f}")
    return exact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--num-dense", type=int, default=6)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    failures = 0
    checked = 0
    with safe_open(args.shard, framework="pt", device="cuda") as f:
        keys = list(f.keys())
        expert_weights = sorted(k for k in keys if ".experts." in k and ".shared_" not in k and k.endswith(".weight"))
        print(f"MXFP4: comparing {min(args.num_experts, len(expert_weights))} of {len(expert_weights)} expert weights")
        for key in expert_weights[: args.num_experts]:
            packed = f.get_tensor(key)
            scale = f.get_tensor(key.removesuffix(".weight") + ".scale")
            if not is_mxfp4_weight(packed, scale):
                print(f"  {key}: not detected as MXFP4, skipped")
                failures += 1
                continue
            checked += 1
            failures += not compare(key, packed, scale)

        dense_weights = sorted(
            k for k in keys if k.endswith(".weight") and k.removesuffix(".weight") + ".scale" in keys
        )
        dense_weights = [k for k in dense_weights if k not in set(expert_weights)]
        print(f"\nblock FP8: comparing {min(args.num_dense, len(dense_weights))} of {len(dense_weights)} dense weights")
        for key in dense_weights[: args.num_dense]:
            weight = f.get_tensor(key)
            scale = f.get_tensor(key.removesuffix(".weight") + ".scale")
            if is_mxfp4_weight(weight, scale):
                print(f"  {key}: unexpectedly detected as MXFP4")
                failures += 1
                continue
            checked += 1
            failures += not compare_block_fp8(key, weight, scale)

    print(f"\n{checked} compared, {failures} mismatched")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
