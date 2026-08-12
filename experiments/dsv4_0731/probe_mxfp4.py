"""Probe the routed-expert tensor layout of a DeepSeek-V4 MXFP4 checkpoint shard.

Verifies that packed FP4 weights and E8M0 scales decode under the OCP MX
convention (32-element blocks, ``E2M1`` elements, low nibble first) and reports
the statistics that distinguish a correct decode from a mis-parsed one.
"""

import argparse

import torch
from safetensors import safe_open
from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

MXFP4_BLOCK = 32
E2M1_MAX = 6.0


def dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode a packed MXFP4 weight and its E8M0 scale to float32."""
    packed_u8 = packed.view(torch.uint8)
    scale_u8 = scale.view(torch.uint8)
    out = MXFP4QuantizeUtil.dequantize(
        packed_u8,
        dtype=torch.float32,
        scale=scale_u8.reshape(-1, 1),
        block_sizes=[MXFP4_BLOCK],
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.reshape(packed.shape[0], packed.shape[1] * 2).float()


def report(name: str, packed: torch.Tensor, scale: torch.Tensor) -> None:
    logical_cols = packed.shape[1] * 2
    groups = scale.shape[1]
    print(f"\n=== {name}")
    print(f"  packed {tuple(packed.shape)} {packed.dtype} -> logical [{packed.shape[0]}, {logical_cols}]")
    print(f"  scale  {tuple(scale.shape)} {scale.dtype}   group size = {logical_cols // groups}")

    w = dequantize_mxfp4(packed, scale)
    blocks = w.reshape(-1, MXFP4_BLOCK)
    exp = scale.view(torch.uint8).reshape(-1).float() - 127.0
    block_amax = blocks.abs().amax(dim=-1)
    # A correct decode leaves every block's amax at E2M1_max times its scale,
    # unless the block is all zeros or its scale saturated at the E8M0 floor.
    ratio = block_amax / torch.exp2(exp)
    nonzero = block_amax > 0
    print(f"  finite: {torch.isfinite(w).all().item()}  nonzero blocks: {nonzero.float().mean():.4f}")
    print(f"  amax/2^s  mean={ratio[nonzero].mean():.4f}  p01={ratio[nonzero].quantile(0.01):.4f}  max={ratio.max():.4f}")
    print(f"  at E2M1 max (ratio==6): {(ratio[nonzero] == E2M1_MAX).float().mean():.4f}")
    print(f"  weight   std={w.std():.6f}  absmax={w.abs().max():.6f}")

    # Re-encoding must reproduce the stored payload bit-for-bit if the decode
    # inverted the exact quantizer that produced the checkpoint.
    requant, rescale = MXFP4QuantizeUtil.quantize(w.reshape(-1, MXFP4_BLOCK), MXFP4_BLOCK)
    requant_data = requant._quantized_data if hasattr(requant, "_quantized_data") else requant[-1]
    same_w = torch.equal(requant_data.reshape(packed.shape).view(torch.uint8), packed.view(torch.uint8))
    same_s = torch.equal(rescale.reshape(-1).to(torch.uint8), scale.view(torch.uint8).reshape(-1))
    print(f"  round-trip identical: weights={same_w} scales={same_s}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--max-tensors", type=int, default=3)
    args = parser.parse_args()

    with safe_open(args.shard, framework="pt", device="cuda") as f:
        keys = list(f.keys())
        expert_weights = sorted(k for k in keys if ".experts." in k and k.endswith(".weight"))
        print(f"shard holds {len(keys)} tensors, {len(expert_weights)} routed-expert weights")
        for key in expert_weights[: args.max_tensors]:
            report(key, f.get_tensor(key), f.get_tensor(key.removesuffix(".weight") + ".scale"))


if __name__ == "__main__":
    main()
