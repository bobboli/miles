import json

from tests.ci.ci_register import register_cpu_ci
from tools.convert_hf_to_mxfp8 import convert_mxfp8

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def test_conversion_marks_experts_as_fp8(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"num_hidden_layers": 1, "expert_dtype": "fp4"}),
        encoding="utf-8",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}}),
        encoding="utf-8",
    )

    convert_mxfp8(str(source), str(destination), device="cpu")

    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    assert config["expert_dtype"] == "fp8"
    assert config["quantization_config"]["quant_method"] == "mxfp8"
