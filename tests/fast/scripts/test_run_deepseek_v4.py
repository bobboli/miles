import json

import pytest

from scripts import run_deepseek_v4
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def _args(tmp_path, **overrides) -> run_deepseek_v4.ScriptArgs:
    values = {
        "model_name": "DeepSeek-V4-Flash-0731",
        "model_dir": str(tmp_path),
        "model_local_dir": str(tmp_path),
        "hardware": "B300",
    }
    values.update(overrides)
    return run_deepseek_v4.ScriptArgs(**values)


def _checkpoint(tmp_path, *, expert_dtype: str | None) -> str:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {} if expert_dtype is None else {"expert_dtype": expert_dtype}
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(checkpoint)


@pytest.mark.parametrize(
    ("expert_dtype", "expected"),
    [("fp4", "fp4"), ("fp8", "fp8"), (None, "fp8")],
)
def test_rollout_expert_dtype_follows_checkpoint_metadata(tmp_path, expert_dtype, expected):
    checkpoint = _checkpoint(tmp_path, expert_dtype=expert_dtype)

    assert run_deepseek_v4._resolve_rollout_expert_dtype(_args(tmp_path), checkpoint) == expected


def test_converted_rollout_ignores_source_expert_metadata(tmp_path):
    checkpoint = _checkpoint(tmp_path, expert_dtype="fp4")
    args = _args(tmp_path, rollout_fp8=False, rollout_mxfp8=True)

    assert run_deepseek_v4._resolve_rollout_expert_dtype(args, checkpoint) == "fp8"


def test_rollout_expert_dtype_override_is_authoritative(tmp_path):
    checkpoint = _checkpoint(tmp_path, expert_dtype="fp4")
    args = _args(tmp_path, rollout_expert_dtype="fp8")

    assert run_deepseek_v4._resolve_rollout_expert_dtype(args, checkpoint) == "fp8"


def test_rollout_expert_dtype_rejects_unknown_metadata(tmp_path):
    checkpoint = _checkpoint(tmp_path, expert_dtype="nvfp4")

    with pytest.raises(ValueError, match="Unsupported expert_dtype='nvfp4'"):
        run_deepseek_v4._resolve_rollout_expert_dtype(_args(tmp_path), checkpoint)
