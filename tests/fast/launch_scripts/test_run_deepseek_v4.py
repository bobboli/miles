import json
from unittest.mock import Mock

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)

_SINGLE_NODE_4LAYER = {
    "hardware": "H200",
    "model_name": "DeepSeek-V4-Flash-FP8-4layer",
    "num_nodes": 1,
    "num_gpus_per_node": 4,
}
_EIGHT_NODES = {"hardware": "H200", "num_nodes": 8, "num_gpus_per_node": 4}
_EIGHT_NODES_OF_8 = {"hardware": "H200", "num_nodes": 8, "num_gpus_per_node": 8}
_THIRTY_TWO_NODES_OF_8 = {"hardware": "H200", "num_nodes": 32, "num_gpus_per_node": 8}


def _train_command(monkeypatch, tmp_path, overrides):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")
    call_entrypoint(module, "train", overrides, sandbox=tmp_path)
    return recording.commands[-1]


@pytest.mark.parametrize(
    ("cp_size", "expected"),
    [
        (None, ["--tensor-model-parallel-size 4", "--sequence-parallel", "--context-parallel-size 1"]),
        (2, ["--tensor-model-parallel-size 2", "--sequence-parallel", "--context-parallel-size 2", "--allgather-cp"]),
        (4, ["--tensor-model-parallel-size 1", "--context-parallel-size 4", "--allgather-cp"]),
    ],
)
def test_single_node_miles_impl_splits_gpus_between_tp_and_cp(monkeypatch, tmp_path, cp_size, expected):
    """DSV4 CP must use the all-gather split; TP takes the GPUs CP leaves, with SP only above TP1."""
    overrides = _SINGLE_NODE_4LAYER | {"dsv4_impl": "miles"} | ({"cp_size": cp_size} if cp_size else {})
    command = _train_command(monkeypatch, tmp_path, overrides)

    for flag in expected:
        assert flag in command
    tp_size = 4 // (cp_size or 1)
    assert ("--allgather-cp" in command) == (tp_size < 4)
    assert ("--sequence-parallel" in command) == (tp_size > 1)


# Every recipe that pins its CP size, with that size.
_FIXED_CP_RECIPES = [
    (_SINGLE_NODE_4LAYER | {"dsv4_impl": "megatron"}, 1),
    (_EIGHT_NODES | {"dsv4_impl": "megatron"}, 1),
    (_EIGHT_NODES | {"dsv4_impl": "miles"}, 2),
    (_EIGHT_NODES_OF_8 | {"dsv4_impl": "megatron"}, 1),
    (_EIGHT_NODES_OF_8 | {"dsv4_impl": "miles"}, 1),
    (_THIRTY_TWO_NODES_OF_8 | {"dsv4_impl": "miles"}, 1),
]


@pytest.mark.parametrize(("recipe", "recipe_cp_size"), _FIXED_CP_RECIPES)
def test_recipes_with_a_fixed_cp_size_accept_it_and_reject_another(monkeypatch, tmp_path, recipe, recipe_cp_size):
    command = _train_command(monkeypatch, tmp_path, recipe | {"cp_size": recipe_cp_size})
    assert f"--context-parallel-size {recipe_cp_size}" in command

    with pytest.raises(NotImplementedError, match="is untested here"):
        _train_command(monkeypatch, tmp_path, recipe | {"cp_size": recipe_cp_size * 2})


def test_a_node_count_without_a_recipe_reports_the_missing_recipe(monkeypatch, tmp_path):
    with pytest.raises(NotImplementedError, match="No pre-set parallel config"):
        _train_command(
            monkeypatch, tmp_path, {"hardware": "H200", "num_nodes": 2, "num_gpus_per_node": 8, "cp_size": 2}
        )


@pytest.mark.parametrize(
    ("overrides", "expected_size"),
    [
        ({"hardware": "H200", "num_nodes": 8, "num_gpus_per_node": 4}, 4),
        ({"hardware": "GB300", "num_nodes": 8, "num_gpus_per_node": 4}, 8),
        (
            {
                "hardware": "H200",
                "model_name": "DeepSeek-V4-Flash-FP8-4layer",
                "num_nodes": 1,
                "num_gpus_per_node": 4,
            },
            4,
        ),
        (
            {
                "hardware": "GB300",
                "model_name": "DeepSeek-V4-Flash-FP8-4layer",
                "num_nodes": 1,
                "num_gpus_per_node": 4,
            },
            4,
        ),
    ],
)
def test_the_rollout_profile_follows_the_hardware(monkeypatch, tmp_path, overrides, expected_size):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")

    call_entrypoint(module, "train", overrides, sandbox=tmp_path)

    train_command = recording.commands[-1]
    assert f"--rollout-num-gpus-per-engine {expected_size}" in train_command
    assert f"--sglang-tp-size {expected_size}" in train_command
    assert f"--sglang-ep-size {expected_size}" in train_command


def _direct_hf_args(tmp_path, *, rollout_precision="mxfp8"):
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")
    return module, module.ScriptArgs(
        model_name="DeepSeek-V4-Flash",
        init_model_source="hf",
        rollout_weight_source="trainer",
        model_dir=str(tmp_path),
        model_local_dir=str(tmp_path),
        hardware="B300",
        train_precision="mxfp8",
        rollout_precision=rollout_precision,
    )


def test_direct_hf_full_train_skips_offline_weight_conversion(tmp_path, monkeypatch):
    module, args = _direct_hf_args(tmp_path)
    prepare_download = Mock()
    prepare_single = Mock()
    prepare_mxfp8 = Mock()
    prepare_spmd = Mock()
    prepare_schema = Mock()
    train = Mock()
    monkeypatch.setattr(module, "_prepare_download", prepare_download)
    monkeypatch.setattr(module, "_prepare_single", prepare_single)
    monkeypatch.setattr(module, "_prepare_mxfp8", prepare_mxfp8)
    monkeypatch.setattr(module, "_prepare_spmd", prepare_spmd)
    monkeypatch.setattr(module, "_prepare_mxfp8_schema", prepare_schema)
    monkeypatch.setattr(module, "_train", train)

    module._full_train(args)

    prepare_download.assert_called_once_with(args)
    prepare_single.assert_not_called()
    prepare_mxfp8.assert_not_called()
    prepare_spmd.assert_not_called()
    prepare_schema.assert_called_once_with(args)
    train.assert_called_once_with(args)


def test_trainer_owned_rollout_uses_dummy_sglang_initialization(tmp_path, monkeypatch):
    module, args = _direct_hf_args(tmp_path, rollout_precision="blockwise-fp8")
    source = tmp_path / "DeepSeek-V4-Flash"
    source.mkdir()
    (source / "config.json").write_text('{"expert_dtype": "fp4"}', encoding="utf-8")
    execute_train = Mock()
    backend = Mock(execute_train=execute_train)
    monkeypatch.setattr(args, "create_backend", lambda: backend)

    module._train(args)

    extra_env_vars = execute_train.call_args.kwargs["extra_env_vars"]
    assert extra_env_vars["MILES_SGLANG_DUMMY_LOAD"] == "1"


def test_trainer_owned_rollout_paths_select_source_and_mxfp8_schema(tmp_path):
    module, mxfp8_args = _direct_hf_args(tmp_path)
    _, mxfp4_args = _direct_hf_args(tmp_path, rollout_precision="blockwise-fp8")
    source = tmp_path / "DeepSeek-V4-Flash"

    assert module._trainer_checkpoint_path(mxfp8_args) == str(source)
    assert module._rollout_checkpoint_path(mxfp8_args) == str(tmp_path / "DeepSeek-V4-Flash-MXFP8-schema")
    assert module._trainer_checkpoint_path(mxfp4_args) == str(source)
    assert module._rollout_checkpoint_path(mxfp4_args) == str(source)


def test_direct_hf_prepare_command_copies_seed_and_schema_to_worker_storage(tmp_path):
    module, args = _direct_hf_args(tmp_path)
    args.model_local_dir = str(tmp_path / "worker")

    command = module._prepare_cmd(args)["trainer"]

    assert str(tmp_path / "DeepSeek-V4-Flash") in command
    assert str(tmp_path / "worker" / "DeepSeek-V4-Flash") in command
    assert str(tmp_path / "DeepSeek-V4-Flash-MXFP8-schema") in command
    assert str(tmp_path / "worker" / "DeepSeek-V4-Flash-MXFP8-schema") in command
    assert args.torch_dist_name not in command
    assert module._prepare_cmd(module.ScriptArgs(hardware="B300")) == {}


@pytest.mark.parametrize("train_precision", ["bf16", "blockwise-fp8", "mxfp8"])
@pytest.mark.parametrize("rollout_precision", ["bf16", "blockwise-fp8", "mxfp8"])
def test_precision_selections_reach_training_recipe_and_rollout_backends(
    monkeypatch, tmp_path, train_precision, rollout_precision
):
    command = _train_command(
        monkeypatch,
        tmp_path,
        _SINGLE_NODE_4LAYER
        | {"hardware": "B300", "train_precision": train_precision, "rollout_precision": rollout_precision},
    )

    assert "--bf16" in command
    if train_precision == "bf16":
        assert "--fp8-recipe" not in command
        assert "--te-precision-config-file" not in command
    else:
        recipe = "blockwise" if train_precision == "blockwise-fp8" else "mxfp8"
        assert f"--fp8-recipe {recipe}" in command
        assert "--te-precision-config-file" in command
    gemm = "flashinfer_cutlass" if rollout_precision == "mxfp8" else "auto"
    moe = "flashinfer_trtllm_routed" if rollout_precision == "mxfp8" else "auto"
    assert f"--sglang-fp8-gemm-backend {gemm}" in command
    assert f"--sglang-moe-runner-backend {moe}" in command
    suffix = {"bf16": "-bf16", "blockwise-fp8": "", "mxfp8": "-MXFP8"}[rollout_precision]
    assert f"--hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8-4layer{suffix} " in command


def test_explicit_auto_delegates_backend_selection_to_sglang(monkeypatch, tmp_path):
    command = _train_command(
        monkeypatch,
        tmp_path,
        _SINGLE_NODE_4LAYER
        | {
            "hardware": "B300",
            "rollout_precision": "mxfp8",
            "sglang_moe_runner_backend": "auto",
            "sglang_fp8_gemm_backend": "auto",
        },
    )
    assert "--sglang-moe-runner-backend auto" in command
    assert "--sglang-fp8-gemm-backend auto" in command


@pytest.mark.parametrize("field", ["train_precision", "rollout_precision"])
def test_mxfp8_requires_blackwell(field):
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")
    with pytest.raises(AssertionError, match="require Blackwell"):
        module.ScriptArgs(hardware="H200", **{field: "mxfp8"})


@pytest.mark.parametrize("train_precision", ["mxfp8", "mxfp4-qat"])
def test_packed_mxfp4_rollout_with_optional_qat(tmp_path, monkeypatch, train_precision):
    module, args = _direct_hf_args(tmp_path, rollout_precision="blockwise-fp8")
    args.train_precision = train_precision
    args.skip_saving = True
    source = tmp_path / "DeepSeek-V4-Flash"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"expert_dtype": "fp4"}), encoding="utf-8")
    execute_train = Mock()
    backend = Mock(execute_train=execute_train)
    monkeypatch.setattr(args, "create_backend", lambda: backend)

    module._train(args)

    train_args = execute_train.call_args.kwargs["train_args"]
    assert "--megatron-to-hf-mode bridge" in train_args
    assert "--dsv4-impl megatron" in train_args
    assert "--qkv-format thd" in train_args
    assert ("--dsv4-mxfp4-qat" in train_args) == (train_precision == "mxfp4-qat")
    assert "--fp8-recipe mxfp8" in train_args
    assert "--rollout-fp4-experts" in train_args
    assert "--sglang-moe-runner-backend flashinfer_mxfp4" in train_args
    assert "--sglang-fp8-gemm-backend auto" in train_args
    assert execute_train.call_args.kwargs["extra_env_vars"]["SGLANG_DSV4_FP4_EXPERTS"] == "1"


def test_mxfp4_qat_requires_blackwell():
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")
    with pytest.raises(AssertionError, match="require Blackwell"):
        module.ScriptArgs(hardware="H200", train_precision="mxfp4-qat")


def test_mxfp4_qat_rejects_rollout_without_packed_experts(tmp_path):
    module, args = _direct_hf_args(tmp_path)
    args.train_precision = "mxfp4-qat"
    with pytest.raises(AssertionError, match="requires packed MXFP4 routed experts"):
        module._train(args)


def test_mxfp4_qat_requires_bridge_model_construction(tmp_path):
    module, args = _direct_hf_args(tmp_path, rollout_precision="blockwise-fp8")
    args.train_precision = "mxfp4-qat"
    args.init_model_source = "torch_dist"
    with pytest.raises(ValueError, match="Megatron-Bridge provider"):
        module._train(args)


@pytest.mark.parametrize(("precision", "expert_dtype"), [("mxfp8", "fp4"), ("bf16", "fp4"), ("bf16", "fp8")])
def test_incompatible_rollout_precision_and_expert_override_are_rejected(tmp_path, precision, expert_dtype):
    module, args = _direct_hf_args(tmp_path, rollout_precision=precision)
    args.rollout_expert_dtype = expert_dtype
    with pytest.raises(ValueError):
        module._resolve_rollout_expert_dtype(args, str(tmp_path))
