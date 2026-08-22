from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

from miles.utils.mxfp4 import E2M1_VALUES, MXFP4_GROUP_SIZE, mxfp4_quantize
from miles_plugins.models.deepseek_v4.ops.mxfp4_qat import (
    _wrap_get_weight_tensors,
    mxfp4_fake_quantize_ste,
    mxfp4_quantize_dequantize,
)

register_cuda_ci(est_time=30, suite="stage-b-2-gpu-h200", labels=["precision"])

E8M0_BIAS = 127


def _dequantize_reference(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logical_shape = (*packed.shape[:-1], packed.shape[-1] * 2)
    codes = packed.view(torch.uint8)
    nibbles = torch.stack((codes & 0x0F, (codes >> 4) & 0x0F), dim=-1).flatten(-2)
    table = torch.tensor(E2M1_VALUES + tuple(-value for value in E2M1_VALUES), device=packed.device)
    values = table[nibbles.long()].reshape(-1, MXFP4_GROUP_SIZE)
    scale_factor = torch.exp2(scale.view(torch.uint8).float() - E8M0_BIAS).reshape(-1, 1)
    return (values * scale_factor).reshape(logical_shape)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mxfp4_qat_matches_rollout_codec(dtype):
    torch.manual_seed(0)
    weight = torch.randn((67, 4 * MXFP4_GROUP_SIZE), device="cuda", dtype=dtype)

    actual = mxfp4_quantize_dequantize(weight)
    packed, scale = mxfp4_quantize(weight)
    expected = _dequantize_reference(packed, scale).to(dtype)

    assert torch.equal(actual, expected)


def test_mxfp4_qat_matches_rollout_codec_at_boundaries_and_zero():
    boundaries = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda")
    row = torch.cat((boundaries, -boundaries, torch.zeros(MXFP4_GROUP_SIZE - 2 * boundaries.numel(), device="cuda")))
    weight = torch.stack((row, torch.zeros_like(row)))

    actual = mxfp4_quantize_dequantize(weight)
    packed, scale = mxfp4_quantize(weight)
    expected = _dequantize_reference(packed, scale)

    assert torch.equal(actual, expected)


def test_mxfp4_qat_ste_passes_gradient_through():
    weight = torch.randn((4, 2 * MXFP4_GROUP_SIZE), device="cuda", requires_grad=True)
    grad = torch.randn_like(weight)

    mxfp4_fake_quantize_ste(weight).backward(grad)

    assert torch.equal(weight.grad, grad)


def test_mxfp4_qat_ste_preserves_main_grad():
    weight = torch.randn((2, MXFP4_GROUP_SIZE), device="cuda")
    weight.main_grad = torch.empty_like(weight)

    output = mxfp4_fake_quantize_ste(weight)

    assert output.main_grad is weight.main_grad


def test_grouped_linear_patch_is_config_gated(monkeypatch):
    weight = torch.randn((2, MXFP4_GROUP_SIZE), device="cuda")
    module = SimpleNamespace(config=SimpleNamespace(dsv4_mxfp4_qat=False))
    wrapped = _wrap_get_weight_tensors(lambda _: [weight])

    assert wrapped(module) == [weight]

    sentinel = torch.zeros_like(weight)
    monkeypatch.setattr(
        "miles_plugins.models.deepseek_v4.ops.mxfp4_qat.mxfp4_fake_quantize_ste",
        lambda _: sentinel,
    )
    module.config.dsv4_mxfp4_qat = True
    assert wrapped(module) == [sentinel]
