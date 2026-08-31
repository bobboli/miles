import json
from unittest.mock import Mock

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


def _direct_hf_args(
    tmp_path,
    *,
    rollout_mxfp8: bool,
    model_name: str = "DeepSeek-V4-Flash-0731",
) -> run_deepseek_v4.ScriptArgs:
    return run_deepseek_v4.ScriptArgs(
        model_name=model_name,
        init_model_source="hf",
        rollout_weight_source="trainer",
        model_dir=str(tmp_path),
        model_local_dir=str(tmp_path),
        hardware="B300",
        train_fp8=False,
        train_mxfp8=True,
        rollout_fp8=not rollout_mxfp8,
        rollout_mxfp8=rollout_mxfp8,
    )


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


def test_direct_hf_full_train_skips_offline_weight_conversion(tmp_path, monkeypatch):
    args = _direct_hf_args(tmp_path, rollout_mxfp8=True)
    prepare_download = Mock()
    prepare_single = Mock()
    prepare_mxfp8 = Mock()
    prepare_spmd = Mock()
    prepare_schema = Mock()
    train = Mock()
    monkeypatch.setattr(run_deepseek_v4, "_prepare_download", prepare_download)
    monkeypatch.setattr(run_deepseek_v4, "_prepare_single", prepare_single)
    monkeypatch.setattr(run_deepseek_v4, "_prepare_mxfp8", prepare_mxfp8)
    monkeypatch.setattr(run_deepseek_v4, "_prepare_spmd", prepare_spmd)
    monkeypatch.setattr(run_deepseek_v4, "_prepare_mxfp8_schema", prepare_schema)
    monkeypatch.setattr(run_deepseek_v4, "_train", train)

    run_deepseek_v4._full_train(args)

    prepare_download.assert_called_once_with(args)
    prepare_single.assert_not_called()
    prepare_mxfp8.assert_not_called()
    prepare_spmd.assert_not_called()
    prepare_schema.assert_called_once_with(args)
    train.assert_called_once_with(args)


def test_trainer_owned_rollout_paths_select_only_source_and_schema(tmp_path):
    p1 = _direct_hf_args(tmp_path, rollout_mxfp8=True)
    p2 = _direct_hf_args(tmp_path, rollout_mxfp8=False)
    source = tmp_path / "DeepSeek-V4-Flash-0731"

    assert run_deepseek_v4._trainer_checkpoint_path(p1) == str(source)
    assert run_deepseek_v4._rollout_checkpoint_path(p1) == str(tmp_path / "DeepSeek-V4-Flash-0731-MXFP8-schema")
    assert run_deepseek_v4._trainer_checkpoint_path(p2) == str(source)
    assert run_deepseek_v4._rollout_checkpoint_path(p2) == str(source)


def test_current_flash_supports_direct_hf_mxfp8_rollout(tmp_path):
    args = run_deepseek_v4.ScriptArgs(
        model_name="DeepSeek-V4-Flash",
        init_model_source="hf",
        rollout_weight_source="trainer",
        model_dir=str(tmp_path),
        model_local_dir=str(tmp_path),
        hardware="B300",
        train_fp8=False,
        train_mxfp8=True,
        rollout_fp8=False,
        rollout_mxfp8=True,
    )

    assert args.model_org == "deepseek-ai"
    assert args.megatron_model_type == "deepseek-v4-flash"
    assert run_deepseek_v4._trainer_checkpoint_path(args) == str(tmp_path / "DeepSeek-V4-Flash")
    assert run_deepseek_v4._rollout_checkpoint_path(args) == str(tmp_path / "DeepSeek-V4-Flash-MXFP8-schema")


def test_p3_enables_mxfp4_qat_for_mxfp8_train_and_fp4_rollout(tmp_path, monkeypatch):
    args = _direct_hf_args(tmp_path, rollout_mxfp8=False)
    args.dsv4_mxfp4_qat = True
    args.skip_saving = True
    source = tmp_path / "DeepSeek-V4-Flash-0731"
    source.mkdir()
    (source / "config.json").write_text('{"expert_dtype":"fp4"}', encoding="utf-8")
    execute_train = Mock()
    monkeypatch.setattr(run_deepseek_v4.U, "execute_train", execute_train)

    run_deepseek_v4._train(args)

    train_args = execute_train.call_args.kwargs["train_args"]
    assert "--dsv4-mxfp4-qat" in train_args
    assert "--dsv4-kv-cache-qat" not in train_args
    assert "--fp8-recipe mxfp8" in train_args
    assert "--rollout-fp4-experts" in train_args
    assert "--sglang-kv-cache-dtype fp8_e4m3" in train_args


def test_p4_adds_kv_cache_qat_to_p3_recipe(tmp_path, monkeypatch):
    args = _direct_hf_args(tmp_path, rollout_mxfp8=False)
    args.dsv4_mxfp4_qat = True
    args.dsv4_kv_cache_qat = True
    args.skip_saving = True
    source = tmp_path / "DeepSeek-V4-Flash-0731"
    source.mkdir()
    (source / "config.json").write_text('{"expert_dtype":"fp4"}', encoding="utf-8")
    execute_train = Mock()
    monkeypatch.setattr(run_deepseek_v4.U, "execute_train", execute_train)

    run_deepseek_v4._train(args)

    train_args = execute_train.call_args.kwargs["train_args"]
    assert "--dsv4-mxfp4-qat" in train_args
    assert "--dsv4-kv-cache-qat" in train_args
    assert "--sglang-kv-cache-dtype fp8_e4m3" in train_args


def test_current_flash_supports_p3_and_p4_qat(tmp_path, monkeypatch):
    args = _direct_hf_args(
        tmp_path,
        rollout_mxfp8=False,
        model_name="DeepSeek-V4-Flash",
    )
    args.dsv4_mxfp4_qat = True
    args.dsv4_kv_cache_qat = True
    args.skip_saving = True
    source = tmp_path / "DeepSeek-V4-Flash"
    source.mkdir()
    (source / "config.json").write_text('{"expert_dtype":"fp4"}', encoding="utf-8")
    execute_train = Mock()
    monkeypatch.setattr(run_deepseek_v4.U, "execute_train", execute_train)

    run_deepseek_v4._train(args)

    train_args = execute_train.call_args.kwargs["train_args"]
    assert "--dsv4-mxfp4-qat" in train_args
    assert "--dsv4-kv-cache-qat" in train_args


def test_kv_cache_qat_rejects_bfloat16_rollout_cache(tmp_path):
    args = _direct_hf_args(tmp_path, rollout_mxfp8=False)
    args.dsv4_kv_cache_qat = True
    args.rollout_kv_cache_dtype = "bfloat16"
    source = tmp_path / "DeepSeek-V4-Flash-0731"
    source.mkdir()
    (source / "config.json").write_text('{"expert_dtype":"fp4"}', encoding="utf-8")

    with pytest.raises(AssertionError, match="requires an FP8 E4M3 rollout cache"):
        run_deepseek_v4._train(args)
