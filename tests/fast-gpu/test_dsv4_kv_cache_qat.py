import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

from miles_plugins.models.deepseek_v4.ops.qat import fp8_simulate, fp8_simulate_qat

register_cuda_ci(est_time=90, suite="stage-b-2-gpu-h200", labels=["precision"])

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _reference_qdq(x: torch.Tensor, block_size: int, scale_format: str) -> torch.Tensor:
    rows = x.float().reshape(-1, x.shape[-1] // block_size, block_size)
    scale = rows.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / FP8_MAX
    if scale_format == "ue8m0":
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quantized = (rows * (1.0 / scale)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (quantized.float() * scale).reshape_as(x).to(x.dtype)


@pytest.mark.parametrize(("block_size", "scale_format"), [(64, "ue8m0")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_kv_cache_qat_matches_dsv4_rollout_codec(block_size, scale_format, dtype):
    torch.manual_seed(0)
    x = torch.randn((13, 4 * block_size), device="cuda", dtype=dtype)

    actual = fp8_simulate(x, block_size, scale_format)
    expected = _reference_qdq(x, block_size, scale_format)

    assert torch.equal(actual, expected)


def _sglang_indexer_round_trip(x: torch.Tensor) -> torch.Tensor:
    from sglang.jit_kernel.dsv4.attn import fused_store_cache

    page_size = 64
    indices = torch.arange(x.shape[0], device=x.device, dtype=torch.int32)
    cache = torch.zeros((1, 132 * page_size), device=x.device, dtype=torch.uint8)
    fused_store_cache(x, cache, indices, page_size=page_size, type="indexer")

    rows = []
    scale_offset = 128 * page_size
    for slot in range(x.shape[0]):
        value = cache[0, slot * 128 : (slot + 1) * 128].view(torch.float8_e4m3fn)
        scale = cache[0, scale_offset + slot * 4 : scale_offset + (slot + 1) * 4].view(torch.float32)
        rows.append((value.float() * scale).to(x.dtype))
    return torch.stack(rows)


def _sglang_mla_round_trip(x: torch.Tensor) -> torch.Tensor:
    from sglang.jit_kernel.dsv4.attn import fused_store_cache

    page_size = 64
    indices = torch.arange(x.shape[0], device=x.device, dtype=torch.int32)
    cache = torch.zeros((1, 65 * 576), device=x.device, dtype=torch.uint8)
    fused_store_cache(x, cache, indices, page_size=page_size, type="flashmla")

    rows = []
    scale_offset = 576 * page_size
    for slot in range(x.shape[0]):
        value_offset = slot * 576
        nope = cache[0, value_offset : value_offset + 448].view(torch.float8_e4m3fn)
        rope = cache[0, value_offset + 448 : value_offset + 576].view(torch.bfloat16)
        scale_bits = cache[0, scale_offset + slot * 8 : scale_offset + slot * 8 + 7]
        scale = torch.exp2(scale_bits.float() - 127).repeat_interleave(64)
        rows.append(torch.cat(((nope.float() * scale).to(x.dtype), rope)))
    return torch.stack(rows)


def test_kv_cache_qat_matches_sglang_cuda_cache_writers():
    pytest.importorskip("sglang")
    torch.manual_seed(1)
    indexer = torch.randn((13, 128), device="cuda", dtype=torch.bfloat16)
    mla = torch.randn((13, 512), device="cuda", dtype=torch.bfloat16)

    expected_indexer = fp8_simulate(indexer, 128, "fp32")
    expected_mla = torch.cat((fp8_simulate(mla[:, :448], 64, "ue8m0"), mla[:, 448:]), dim=-1)

    assert torch.equal(_sglang_indexer_round_trip(indexer), expected_indexer)
    assert torch.equal(_sglang_mla_round_trip(mla), expected_mla)


@pytest.mark.parametrize(("block_size", "scale_format"), [(64, "ue8m0"), (128, "fp32")])
def test_kv_cache_qat_matches_codec_for_zero_and_tiny_values(block_size, scale_format):
    tiny = torch.finfo(torch.bfloat16).tiny
    values = torch.tensor([0.0, tiny, -tiny, 1e-5, -1e-5, 1e-4, -1e-4], device="cuda")
    repeats = (2 * block_size + values.numel() - 1) // values.numel()
    x = values.repeat(repeats)[: 2 * block_size].reshape(2, block_size).to(torch.bfloat16)

    actual = fp8_simulate(x, block_size, scale_format)
    expected = _reference_qdq(x, block_size, scale_format)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("scale_format", ["ue8m0", "fp32"])
def test_kv_cache_qat_ste_passes_gradient_through(scale_format):
    x = torch.randn((4, 128), device="cuda", requires_grad=True)
    grad = torch.randn_like(x)

    fp8_simulate_qat(x, 128, scale_format).backward(grad)

    assert torch.equal(x.grad, grad)


def test_kv_cache_qat_rejects_unknown_scale_format():
    x = torch.zeros((1, 128), device="cuda")

    with pytest.raises(ValueError, match="Unsupported FP8 scale format"):
        fp8_simulate(x, 128, "unknown")
