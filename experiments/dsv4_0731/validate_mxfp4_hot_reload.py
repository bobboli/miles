"""Check that an online weight update reproduces the initial load for MXFP4 experts.

The rollout backend builds a kernel layout after every load, and that layout
differs from the layout weights arrive in — in dtype for all four expert
parameters, and in extent for the second gemm's scale. Nothing in the loading
path rejects a mismatch: a scale byte of 130 written into the kernel layout's
``float8_e4m3fn`` parameter is cast to 112 and the model serves wrong weights.

This drives the contract directly on a small layer, so a break is a one-minute
failure here rather than a wrong number after twenty-five minutes on eight nodes.
Run it inside the image with the MXFP4 patch applied:

    patch --batch --forward -p1 -d /sgl-workspace/sglang < patches/mxfp4_trtllm_hot_reload.patch
    python experiments/dsv4_0731/validate_mxfp4_hot_reload.py
"""

from __future__ import annotations

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    Mxfp4FlashinferTrtllmMoEMethod,
    restore_moe_load_layout,
)

EXPERT_PARAMS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
)
NUM_EXPERTS = 2
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 64


class _NoOpFp8:
    """The FP8 base method's own post-load pass, which this contract does not cover."""

    def process_weights_after_loading(self, layer: Module) -> None:
        pass


def build_layer() -> tuple[Module, Mxfp4FlashinferTrtllmMoEMethod]:
    layer = Module().to("cuda")
    layer.num_local_experts = NUM_EXPERTS
    method = Mxfp4FlashinferTrtllmMoEMethod.__new__(Mxfp4FlashinferTrtllmMoEMethod)
    method._fp8 = _NoOpFp8()
    method.prefix = "validate"
    method._kernel_layout = {}
    layer.quant_method = method
    method.create_weights(
        layer,
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        torch.bfloat16,
        alloc_device="cuda",
        weight_loader=lambda *args, **kwargs: None,
    )
    return layer, method


def load_weights(layer: Module) -> None:
    """Stand in for the loader writing packed MXFP4 payloads and UE8M0 scales."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    for name, param in layer.named_parameters():
        if "scale" in name:
            biased = torch.randint(
                100, 140, param.shape, generator=generator, device="cuda", dtype=torch.uint8
            )
            param.data.copy_(biased.view(torch.float8_e8m0fnu))
        else:
            param.data.copy_(
                torch.randint(
                    -128, 127, param.shape, generator=generator, device="cuda", dtype=torch.int8
                )
            )


def main() -> None:
    layer, method = build_layer()
    load_layout = {n: (tuple(p.shape), p.dtype) for n, p in layer.named_parameters()}

    load_weights(layer)
    method.process_weights_after_loading(layer)
    initial = {n: p.data.clone() for n, p in layer.named_parameters()}
    addresses = {n: p.data.data_ptr() for n, p in layer.named_parameters()}

    restore_moe_load_layout(layer)
    restored = {n: (tuple(p.shape), p.dtype) for n, p in layer.named_parameters()}
    assert restored == load_layout, f"restore did not reach the load layout: {restored}"
    for name in EXPERT_PARAMS:
        param = getattr(layer, name)
        assert hasattr(param, "weight_loader"), f"{name} lost its weight_loader"

    load_weights(layer)
    method.process_weights_after_loading(layer)

    for name, param in layer.named_parameters():
        # A captured CUDA graph reads the address it saw, so a rebuild that lands
        # anywhere else serves the previous weights however equal the values look.
        assert (
            param.data.data_ptr() == addresses[name]
        ), f"{name}: kernel-layout storage moved across an update"
        before = initial[name]
        assert before.shape == param.shape, f"{name}: {before.shape} became {param.shape}"
        assert torch.equal(
            before.view(torch.uint8), param.data.view(torch.uint8)
        ), f"{name}: an update does not reproduce the initial load"

    print(
        f"MXFP4 hot reload reproduces the initial load, in place, on "
        f"{len(EXPERT_PARAMS)} parameters."
    )


if __name__ == "__main__":
    main()
