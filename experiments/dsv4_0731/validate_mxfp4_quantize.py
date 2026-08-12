"""Check the MXFP4 encoder against real checkpoint bytes.

``validate_mxfp4_dequant.py`` covers the decode direction against SGLang's own
cast. This covers the direction the weight updater uses: values that came out of
a release expert tensor, re-encoded by ``mxfp4_quantize``, must reproduce the
tensor they came from — payload and scale alike. The encoding is not unique in
general, so a synthetic round trip only pins down the values; comparing against a
checkpoint pins down the bytes, and with them the nibble order and the block
exponent convention the kernels read.

Run it inside the image against a downloaded shard:

    python experiments/dsv4_0731/validate_mxfp4_quantize.py refdata/shard02.safetensors
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

from fp8_cast_bf16 import is_mxfp4_weight, mxfp4_dequant  # noqa: E402
from miles.utils.mxfp4 import mxfp4_quantize  # noqa: E402

SCALE_SUFFIX = ".scale"
WEIGHT_SUFFIX = ".weight"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", help="A safetensors shard from the release checkpoint.")
    parser.add_argument("--limit", type=int, default=8, help="Expert tensors to check.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checked = 0
    with safe_open(args.shard, framework="pt", device=args.device) as shard:
        names = set(shard.keys())
        for name in sorted(names):
            if ".experts." not in name or not name.endswith(WEIGHT_SUFFIX):
                continue
            scale_name = name[: -len(WEIGHT_SUFFIX)] + SCALE_SUFFIX
            if scale_name not in names:
                continue
            weight, scale = shard.get_tensor(name), shard.get_tensor(scale_name)
            if not is_mxfp4_weight(weight, scale):
                continue

            values = mxfp4_dequant(weight, scale)
            packed, requantized_scale = mxfp4_quantize(values)

            assert torch.equal(
                packed.view(torch.uint8), weight.view(torch.uint8)
            ), f"{name}: re-encoded payload differs from the checkpoint"
            assert torch.equal(
                requantized_scale.view(torch.uint8), scale.view(torch.uint8)
            ), f"{name}: re-encoded scale differs from the checkpoint"
            assert requantized_scale.dtype == scale.dtype, (
                f"{name}: the updater emits {requantized_scale.dtype} where the "
                f"checkpoint stores {scale.dtype}"
            )

            checked += 1
            if checked >= args.limit:
                break

    if checked == 0:
        raise SystemExit(f"No packed MXFP4 expert tensors found in {args.shard}")
    print(f"{checked} expert tensors re-encode to their checkpoint bytes exactly.")


if __name__ == "__main__":
    main()
