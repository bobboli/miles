import json
from types import SimpleNamespace

import pytest

from miles.utils.data import Dataset, _apply_prompt_template
from miles.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


class _Tokenizer:
    name_or_path = ""

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return messages[-1]["content"]


def test_prompt_template_wraps_string_prompt():
    assert _apply_prompt_template("What is 1 + 1?", "Solve this:\n{prompt}\nUse a box.") == (
        "Solve this:\nWhat is 1 + 1?\nUse a box."
    )


def test_prompt_template_wraps_last_user_message_without_mutating_input():
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow-up"},
    ]

    templated = _apply_prompt_template(messages, "Instruction\n{prompt}")

    assert templated[-1]["content"] == "Instruction\nFollow-up"
    assert templated[0]["content"] == "First question"
    assert messages[-1]["content"] == "Follow-up"


@pytest.mark.parametrize("template", ["No placeholder", "{prompt} and {prompt}"])
def test_prompt_template_requires_exactly_one_placeholder(template):
    with pytest.raises(ValueError, match="exactly one"):
        _apply_prompt_template("question", template)

    with pytest.raises(ValueError, match="exactly one"):
        EvalDatasetConfig(name="aime", path="aime.jsonl", prompt_template=template)


def test_prompt_template_can_be_set_in_eval_defaults():
    args = SimpleNamespace()

    datasets = build_eval_dataset_configs(
        args,
        [{"name": "aime", "path": "aime.jsonl"}],
        {"prompt_template": "Answer in a box:\n{prompt}"},
    )

    assert datasets[0].prompt_template == "Answer in a box:\n{prompt}"
    assert datasets[0].prompt_template in datasets[0].cache_key


def test_dataset_applies_prompt_template_before_chat_template(tmp_path):
    path = tmp_path / "aime.jsonl"
    path.write_text(json.dumps({"prompt": "What is 1 + 1?", "label": "2"}) + "\n", encoding="utf-8")
    tokenizer = _Tokenizer()

    dataset = Dataset(
        path=str(path),
        tokenizer=tokenizer,
        processor=None,
        max_length=None,
        prompt_key="prompt",
        label_key="label",
        prompt_template="Solve carefully:\n{prompt}\nReturn \\boxed{}.",
        apply_chat_template=True,
    )

    assert tokenizer.messages == [{"role": "user", "content": "Solve carefully:\nWhat is 1 + 1?\nReturn \\boxed{}."}]
    assert dataset.samples[0].prompt == "Solve carefully:\nWhat is 1 + 1?\nReturn \\boxed{}."
    assert dataset.samples[0].label == "2"
