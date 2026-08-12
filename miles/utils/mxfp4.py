import torch


MXFP4_GROUP_SIZE = 32
E8M0_BIAS = 127
E2M1_MAX = 6.0

# Magnitudes an E2M1 element can take, indexed by its three low bits. The
# midpoints between them decide which code a value rounds to.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def mxfp4_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to MXFP4 with one UE8M0 exponent per 32 elements.

    Returns the payload packed two elements per byte, low nibble first, and the
    biased exponents. The scale is left as ``uint8`` to match what the MXFP8
    quantizer hands the weight updater; checkpoints on disk label the same bytes
    ``float8_e8m0fnu``, and readers view rather than convert them.
    """
    weight = weight.contiguous()
    k = weight.shape[-1]
    if k % MXFP4_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {k} must be divisible by {MXFP4_GROUP_SIZE} for MXFP4.")

    blocks = weight.float().reshape(-1, MXFP4_GROUP_SIZE)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    # A zero block would take log2(0); the E8M0 floor is the smallest exponent
    # the format can carry, and it encodes those blocks as all-zero anyway.
    exponent = torch.ceil(torch.log2(amax / E2M1_MAX).clamp(min=-float(E8M0_BIAS)))
    exponent = torch.where(amax > 0, exponent, torch.full_like(exponent, -float(E8M0_BIAS)))

    scaled = blocks / torch.exp2(exponent)
    magnitude = (scaled.abs().unsqueeze(-1) > _E2M1_BOUNDS.to(weight.device)).sum(dim=-1)
    codes = (magnitude + torch.signbit(scaled).to(magnitude.dtype) * 0b1000).to(torch.uint8)

    codes = codes.reshape(*weight.shape[:-1], k)
    packed = (codes[..., 1::2] << 4) | codes[..., 0::2]
    scale = (exponent + E8M0_BIAS).to(torch.uint8).reshape(*weight.shape[:-1], k // MXFP4_GROUP_SIZE)
    return packed.view(torch.int8), scale
