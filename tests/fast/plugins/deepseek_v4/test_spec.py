import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

from miles_plugins.models.deepseek_v4.deepseek_v4 import get_dsv4_spec


def test_dsv4_spec_import_does_not_require_optional_gpu_kernels():
    code = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def import_without_optional_gpu_kernels(name, *args, **kwargs):
            blocked_packages = ("tile_kernels", "tilelang")
            if any(name == package or name.startswith(f"{package}.") for package in blocked_packages):
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = import_without_optional_gpu_kernels
        import miles_plugins.models.deepseek_v4.deepseek_v4
        """
    )

    subprocess.run([sys.executable, "-c", code], check=True)


def test_get_dsv4_spec_applies_plugin_runtime_config():
    args = SimpleNamespace(
        miles_dsa_topk_backend="torch",
        dsv4_o_groups=8,
        dsv4_o_lora_rank=1024,
        dsv4_window_size=128,
        dsv4_compress_ratios=[0, 0, 4],
        dsv4_compress_rope_theta=160000,
        dsv4_hc_mult=4,
        dsv4_hc_sinkhorn_iters=20,
        dsv4_hc_eps=1e-6,
    )
    config = SimpleNamespace(experimental_attention_variant="dsv4")

    with patch(
        "miles_plugins.models.deepseek_v4.deepseek_v4.get_transformer_block_with_experimental_attention_variant_spec",
        return_value="layer-spec",
    ):
        assert get_dsv4_spec(args, config, vp_stage=None) == "layer-spec"

    assert config.miles_dsa_topk_backend == "torch"
    assert config.dsv4_o_groups == 8
    assert config.dsv4_o_lora_rank == 1024
    assert config.dsv4_window_size == 128
    assert config.dsv4_compress_ratios == [0, 0, 4]
    assert config.dsv4_compress_rope_theta == 160000
    assert config.dsv4_hc_mult == 4
    assert config.dsv4_hc_sinkhorn_iters == 20
    assert config.dsv4_hc_eps == 1e-6
