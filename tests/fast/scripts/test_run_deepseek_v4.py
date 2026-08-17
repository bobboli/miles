from unittest.mock import Mock

from tests.ci.ci_register import register_cpu_ci

from scripts import run_deepseek_v4


register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def _direct_hf_args(tmp_path, *, rollout_mxfp8: bool) -> run_deepseek_v4.ScriptArgs:
    return run_deepseek_v4.ScriptArgs(
        model_name="DeepSeek-V4-Flash-0731",
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
