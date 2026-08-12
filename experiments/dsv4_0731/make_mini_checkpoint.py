"""Build a one-shard DeepSeek-V4 checkpoint from a slice of a real shard.

The slice keeps a few routed experts, the dense attention projections, and the
shared expert, which is enough to exercise every branch of the BF16 cast on the
native tensor layout without materializing the full checkpoint.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_NAME = "model-00001-of-00001.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-experts", type=int, default=2)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keep_prefixes = tuple(f"layers.0.ffn.experts.{i}." for i in range(args.num_experts))
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(args.shard, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(keep_prefixes) or ".experts." not in key:
                tensors[key] = f.get_tensor(key)

    # `embed.weight` marks a native-format V4 checkpoint and drives the HF name
    # remap; without it the slice would be cast under identity names.
    config = json.loads(Path(args.config).read_text())
    tensors["embed.weight"] = torch.zeros(
        config["vocab_size"], config["hidden_size"], dtype=torch.bfloat16
    )

    save_file(tensors, out / SHARD_NAME, metadata={"format": "pt"})
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {k: SHARD_NAME for k in tensors}}, indent=2)
    )
    shutil.copyfile(args.config, out / "config.json")

    dtypes: dict[str, int] = {}
    for tensor in tensors.values():
        dtypes[str(tensor.dtype)] = dtypes.get(str(tensor.dtype), 0) + 1
    print(f"wrote {len(tensors)} tensors to {out}")
    print(f"dtypes: {dtypes}")


if __name__ == "__main__":
    main()
