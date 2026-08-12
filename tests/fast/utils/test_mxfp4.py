import sys
from pathlib import Path

import torch

from miles.utils.mxfp4 import MXFP4_GROUP_SIZE, E2M1_VALUES, mxfp4_quantize

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from fp8_cast_bf16 import mxfp4_dequant  # noqa: E402


def test_quantize_is_exact_on_representable_values():
    # Every E2M1 magnitude at unit scale, positive row then negative row.
    row = torch.tensor(E2M1_VALUES).repeat(MXFP4_GROUP_SIZE // 8)
    weight = torch.stack([row, -row])

    packed, scale = mxfp4_quantize(weight)

    assert torch.equal(mxfp4_dequant(packed, scale).float(), weight)


def test_quantize_round_trips_through_the_shipped_decoder():
    torch.manual_seed(0)
    weight = torch.randn(64, 4 * MXFP4_GROUP_SIZE) * 0.05

    packed, scale = mxfp4_quantize(weight)
    decoded = mxfp4_dequant(packed, scale).float()

    # Re-encoding preserves the values but not necessarily the payload: a block
    # whose amax falls below the top of the E2M1 range can be re-expressed with
    # a smaller exponent and larger codes.
    repacked, rescale = mxfp4_quantize(decoded)
    assert torch.equal(mxfp4_dequant(repacked, rescale).float(), decoded)


def test_quantize_keeps_error_within_the_grid_spacing():
    torch.manual_seed(0)
    weight = torch.randn(32, 2 * MXFP4_GROUP_SIZE)

    packed, scale = mxfp4_quantize(weight)
    decoded = mxfp4_dequant(packed, scale).float()

    blocks = weight.reshape(-1, MXFP4_GROUP_SIZE)
    # The coarsest gap in the E2M1 grid is 2 at the top of the range, halved for
    # round-to-nearest and scaled by each block's own exponent.
    exponent = scale.view(torch.uint8).float().reshape(-1, 1) - 127.0
    tolerance = torch.exp2(exponent)
    assert ((decoded.reshape(-1, MXFP4_GROUP_SIZE) - blocks).abs() <= tolerance).all()


def test_quantize_emits_the_packed_layout():
    weight = torch.zeros(1, MXFP4_GROUP_SIZE)
    weight[0, 0] = 1.0  # code 2 in the low nibble
    weight[0, 1] = 6.0  # code 7 in the high nibble

    packed, scale = mxfp4_quantize(weight)

    assert packed.shape == (1, MXFP4_GROUP_SIZE // 2)
    assert scale.shape == (1, 1)
    assert packed.view(torch.uint8)[0, 0].item() == (7 << 4) | 2


def test_quantize_emits_scales_that_carry_their_value():
    # The rollout backend stages scales in a float parameter, so a scale has to
    # read back as its power of two rather than as its biased exponent.
    weight = torch.zeros(1, MXFP4_GROUP_SIZE)
    weight[0, 0] = 48.0  # 6.0 * 2**3, so the block exponent is 3

    _, scale = mxfp4_quantize(weight)

    assert scale.dtype == torch.float8_e8m0fnu
    assert scale.float().item() == 8.0
    assert scale.view(torch.uint8).item() == 3 + 127


def test_quantize_encodes_zero_blocks_without_nan():
    packed, scale = mxfp4_quantize(torch.zeros(2, MXFP4_GROUP_SIZE))

    decoded = mxfp4_dequant(packed, scale).float()
    assert torch.equal(decoded, torch.zeros(2, MXFP4_GROUP_SIZE))
