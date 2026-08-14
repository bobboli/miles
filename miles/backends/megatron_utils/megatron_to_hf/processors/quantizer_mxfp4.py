import re

from miles.utils.mxfp4 import mxfp4_quantize

# The official DeepSeek-V4 release quantizes only its routed experts to MXFP4;
# attention and the shared expert stay on block-scaled FP8, so this processor
# claims the routed experts and leaves the rest to the FP8 path.
_ROUTED_EXPERT_LINEARS = ("linear_fc1", "linear_fc2")


def is_routed_expert_param(megatron_name: str) -> bool:
    """Whether a Megatron parameter name addresses a routed expert's linear."""
    match = re.search(r"(?:decoder|mtp)\.layers\.\d+\.(.+)", megatron_name)
    if not match:
        return False
    rest = match.groups()[0].replace("transformer_layer.", "")
    expert_match = re.match(r"mlp\.experts\.(.+)\.weight(\d+)", rest)
    return bool(expert_match) and expert_match.groups()[0] in _ROUTED_EXPERT_LINEARS


def quantize_params_mxfp4(converted_named_params):
    """Quantize routed-expert weights to packed MXFP4 with UE8M0 block scales."""
    quantize_named_params = []
    for converted_name, param in converted_named_params:
        # Scales carried alongside the source weights are re-derived here.
        if converted_name.endswith("_scale"):
            continue
        quantize_named_params.extend(_quantize_param(converted_name, param))
    return quantize_named_params


def _quantize_param(name, weight):
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    qweight, scale = mxfp4_quantize(weight)
    return [(name, qweight), (name.replace(".weight", ".weight_scale_inv"), scale)]
