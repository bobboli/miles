"""Fail if a checkpoint still holds quantized payloads or orphaned block scales.

A cast that skips a tensor leaves it byte-identical to the source, which is
indistinguishable from success until the weight is used. Checking the emitted
headers catches that before hours of downstream conversion.
"""

import argparse
import json
import struct
from pathlib import Path

QUANTIZED_DTYPES = {"I8", "U8", "F8_E4M3", "F8_E5M2", "F8_E8M0", "F4"}


def read_header(path: Path) -> dict:
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(length))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    offenders: list[str] = []
    total = 0
    for shard in sorted(model_dir.glob("*.safetensors")):
        for name, meta in read_header(shard).items():
            if name == "__metadata__":
                continue
            total += 1
            if meta["dtype"] in QUANTIZED_DTYPES:
                offenders.append(f"{name}: {meta['dtype']} in {shard.name}")
            elif name.endswith(".scale") or name.endswith("_scale_inv"):
                offenders.append(f"{name}: orphaned block scale in {shard.name}")

    print(f"checked {total} tensors across {len(list(model_dir.glob('*.safetensors')))} shards")
    if offenders:
        print(f"{len(offenders)} unconverted tensors:")
        for line in offenders[:20]:
            print(f"  {line}")
        raise SystemExit(1)
    print("fully dequantized")


if __name__ == "__main__":
    main()
