import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from fp8_cast_bf16 import (  # noqa: E402
    E2M1_VALUES,
    MXFP4_BLOCK_SIZE,
    is_mxfp4_weight,
    mxfp4_dequant,
    to_float_scale,
)

E8M0_BIAS = 127


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes two per byte, low nibble first, as stored on disk."""
    return ((codes[..., 1::2] << 4) | codes[..., 0::2]).to(torch.uint8).view(torch.int8)


def _e8m0(exponents: torch.Tensor) -> torch.Tensor:
    return (exponents + E8M0_BIAS).to(torch.uint8).view(torch.float8_e8m0fnu)


def test_mxfp4_dequant_decodes_value_table():
    # One block per row holding every E2M1 code, positive row then negative row.
    magnitudes = torch.arange(8).repeat(MXFP4_BLOCK_SIZE // 8)
    codes = torch.stack([magnitudes, magnitudes | 0b1000])
    packed = _pack_nibbles(codes)
    scale = _e8m0(torch.zeros(2, 1, dtype=torch.int64))

    out = mxfp4_dequant(packed, scale).float()

    expected = torch.tensor(E2M1_VALUES).repeat(MXFP4_BLOCK_SIZE // 8)
    assert torch.equal(out[0], expected)
    assert torch.equal(out[1], -expected)


def test_mxfp4_dequant_applies_block_exponents():
    codes = torch.full((1, 2 * MXFP4_BLOCK_SIZE), 7, dtype=torch.int64)  # E2M1 max
    packed = _pack_nibbles(codes)
    scale = _e8m0(torch.tensor([[-1, 3]]))

    out = mxfp4_dequant(packed, scale).float()

    assert torch.equal(out[0, :MXFP4_BLOCK_SIZE], torch.full((MXFP4_BLOCK_SIZE,), 3.0))
    assert torch.equal(out[0, MXFP4_BLOCK_SIZE:], torch.full((MXFP4_BLOCK_SIZE,), 48.0))


def test_mxfp4_dequant_preserves_nibble_order():
    codes = torch.tensor([[2, 4] * (MXFP4_BLOCK_SIZE // 2)])  # alternating 1.0, 2.0
    out = mxfp4_dequant(_pack_nibbles(codes), _e8m0(torch.zeros(1, 1, dtype=torch.int64))).float()

    assert out[0, 0] == 1.0
    assert out[0, 1] == 2.0


@pytest.mark.parametrize(
    ("packed_cols", "scale_cols", "expected"),
    [
        (2048, 128, True),  # MXFP4: 4096 logical columns over 32-element blocks
        (4096, 32, False),  # block FP8: 4096 columns over 128-element blocks
    ],
)
def test_is_mxfp4_weight_discriminates_layouts(packed_cols, scale_cols, expected):
    weight = torch.zeros(256, packed_cols, dtype=torch.int8)
    scale = torch.zeros(256, scale_cols)
    assert is_mxfp4_weight(weight, scale) is expected


def test_to_float_scale_decodes_ue8m0_and_passes_through_float32():
    assert torch.equal(to_float_scale(_e8m0(torch.tensor([[-2, 0, 5]]))), torch.tensor([[0.25, 1.0, 32.0]]))

    float_scale = torch.tensor([[0.5, 2.0]])
    assert torch.equal(to_float_scale(float_scale), float_scale)
