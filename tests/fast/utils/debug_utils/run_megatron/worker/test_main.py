from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from megatron.core.enums import ModelType

from miles.utils.debug_utils.run_megatron.worker.main import _build_and_load_model


@pytest.mark.parametrize("run_backward", [False, True])
def test_build_model_wraps_with_ddp_only_for_backward(run_backward):
    args = SimpleNamespace(load=None)
    script = SimpleNamespace(role="actor", run_backward=run_backward)
    provider = Mock()
    model = Mock()

    with (
        patch(
            "miles.utils.debug_utils.run_megatron.worker.main.get_model_provider_func",
            return_value=provider,
        ),
        patch("miles.utils.debug_utils.run_megatron.worker.main.get_model", return_value=[model]) as get_model,
    ):
        assert _build_and_load_model(args, script) == [model]

    get_model.assert_called_once_with(
        provider,
        ModelType.encoder_or_decoder,
        wrap_with_ddp=run_backward,
    )
    model.train.assert_called_once_with()
