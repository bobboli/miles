from typing import Literal

import torch


ScaleFormat = Literal["ue8m0", "fp32"]


def fp8_simulate(x: torch.Tensor, block_size: int, scale_format: ScaleFormat = "ue8m0"):
    """Simulate a per-token FP8 E4M3 cache round trip.

    Both the cast (via :func:`act_quant`) and the cast-back step are routed
    through ``deepseek-ai/TileKernels`` so we share the same FP8 kernels with
    the rest of the DeepSeek stack. DeepSeek V4's MLA cache uses UE8M0 scales,
    while its indexer cache stores an ordinary FP32 scale per 128 elements.
    """
    from tile_kernels.quant import per_token_cast_back

    from miles_plugins.models.deepseek_v4.ops.kernel.act_quant import act_quant

    if scale_format not in {"ue8m0", "fp32"}:
        raise ValueError(f"Unsupported FP8 scale format: {scale_format}")

    x_c = x.contiguous()
    scale_fmt = "ue8m0" if scale_format == "ue8m0" else None
    y, scale = act_quant(x_c, block_size, scale_fmt)

    N = x_c.size(-1)
    y_flat = y.view(-1, N)
    scale_flat = scale.reshape(y_flat.size(0), N // block_size).contiguous()

    out_flat = per_token_cast_back((y_flat, scale_flat), "bf16" if x.dtype == torch.bfloat16 else "fp32", block_size)
    return out_flat.view_as(x_c).to(x.dtype)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128, scale_format="ue8m0"):
        return fp8_simulate(kv, block_size, scale_format)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None, None


def fp8_simulate_qat(
    x: torch.Tensor,
    block_size: int,
    scale_format: ScaleFormat = "ue8m0",
) -> torch.Tensor:
    return DeepSeekV4LinearQATFunc.apply(x, block_size, scale_format)
