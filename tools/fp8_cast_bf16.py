# Adapt from https://github.com/alibaba/Pai-Megatron-Patch/blob/2b201af08336dea0403df7c6b497c964cf5a2e75/toolkits/model_checkpoints_convertor/deepseek/fp8_cast_bf16.py
import json
import os
from argparse import ArgumentParser
from glob import glob

import torch
import triton
import triton.language as tl
from param_name_remap import get_param_name_remap
from safetensors.torch import load_file, save_file
from tqdm import tqdm


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


def to_float_scale(s: torch.Tensor) -> torch.Tensor:
    """Materialize a block scale as float32 whether it is stored as UE8M0 or float."""
    if s.dtype == getattr(torch, "float8_e8m0fnu", None):
        return s.float().contiguous()
    return s.float().contiguous() if s.dtype != torch.float32 else s.contiguous()


# MXFP4 stores two E2M1 elements per byte, low nibble first, with one shared
# UE8M0 exponent per MXFP4_BLOCK_SIZE elements along the input dimension.
MXFP4_BLOCK_SIZE = 32
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_TABLE = torch.tensor(E2M1_VALUES + tuple(-v for v in E2M1_VALUES), dtype=torch.float32)


def is_mxfp4_weight(x: torch.Tensor, s: torch.Tensor) -> bool:
    """Tell a packed MXFP4 weight from a block-scaled FP8 one by its scale extent."""
    return x.dim() == 2 and s.dim() == 2 and s.size(-1) == x.size(-1) * 2 // MXFP4_BLOCK_SIZE


def mxfp4_dequant(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Unpack a rowwise MXFP4 weight and apply its UE8M0 block exponents."""
    assert x.dim() == 2 and s.dim() == 2
    out_dim, packed_dim = x.size()
    in_dim = packed_dim * 2
    assert s.size(0) == out_dim and s.size(1) == in_dim // MXFP4_BLOCK_SIZE

    codes = x.view(torch.uint8)
    nibbles = torch.stack((codes & 0x0F, (codes >> 4) & 0x0F), dim=-1).flatten(1)
    values = E2M1_TABLE.to(x.device)[nibbles.long()]

    exponents = torch.exp2(s.view(torch.uint8).float() - 127.0)
    values = values.view(out_dim, -1, MXFP4_BLOCK_SIZE) * exponents.unsqueeze(-1)
    return values.view(out_dim, in_dim).to(torch.get_default_dtype())


def main(fp8_path, bf16_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    os.system("cp -rf " + fp8_path + "/config.json " + bf16_path)
    os.system("cp -rf " + fp8_path + "/*.py " + bf16_path)
    os.system("cp -rf " + fp8_path + "/tokenizer* " + bf16_path)
    os.system("cp -rf " + fp8_path + "/chat_template* " + bf16_path)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file) as f:
        model_index = json.load(f)
    weight_map_raw = model_index["weight_map"]

    remap = get_param_name_remap(os.path.join(fp8_path, "config.json"), weight_map_raw)
    weight_map_renamed = {}
    raw_name_by_renamed = {}
    for raw_name, file_name in weight_map_raw.items():
        renamed_name = remap(raw_name)
        assert renamed_name not in raw_name_by_renamed, (
            f"Remapped tensor name collision: {renamed_name} from "
            f"{raw_name} and {raw_name_by_renamed[renamed_name]}"
        )
        weight_map_renamed[renamed_name] = file_name
        raw_name_by_renamed[renamed_name] = raw_name

    # Cache for loaded safetensor files
    loaded_files = {}

    # Helper function to get tensor from the correct file
    def get_tensor(raw_tensor_name):
        file_name = weight_map_raw[raw_tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")

        return loaded_files[file_name][raw_tensor_name]

    def raw_scale_name(raw_weight_name):
        """Locate the block scale stored alongside a quantized weight.

        Resolution happens in the checkpoint's own namespace so that tensors the
        HF remap does not recognize still find their scale. Native checkpoints
        name it `<prefix>.scale`; HF-format ones append `_scale_inv`.
        """
        if not raw_weight_name.endswith(".weight"):
            return None
        stem = raw_weight_name.removesuffix(".weight")
        for candidate in (f"{stem}.scale", f"{raw_weight_name}_scale_inv"):
            if candidate in weight_map_raw:
                return candidate
        return None

    scale_names_raw = {scale for name in weight_map_raw if (scale := raw_scale_name(name)) is not None}

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        print(f"Handling file: {safetensor_file}")
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name_raw, weight in current_state_dict.items():
            if weight_name_raw in scale_names_raw:
                continue

            weight_name = remap(weight_name_raw)
            if weight.element_size() == 1:  # FP8 or packed MXFP4 weight
                scale_raw = raw_scale_name(weight_name_raw)
                if scale_raw is None:
                    # Copying a quantized payload through unchanged would look
                    # like a successful cast while producing garbage weights.
                    raise KeyError(
                        f"No block scale found for quantized weight {weight_name_raw} "
                        f"(dtype {weight.dtype}); refusing to emit an unconverted tensor."
                    )
                scale_inv = get_tensor(scale_raw)
                if is_mxfp4_weight(weight, scale_inv):
                    new_state_dict[weight_name] = mxfp4_dequant(weight, scale_inv)
                else:
                    new_state_dict[weight_name] = weight_dequant(weight, to_float_scale(scale_inv))
            else:
                new_state_dict[weight_name] = weight

        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()

    # Update model index: the cast folds every block scale into its weight.
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for scale_raw in scale_names_raw:
        weight_map_renamed.pop(remap(scale_raw), None)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map_renamed}, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
