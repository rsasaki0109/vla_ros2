from __future__ import annotations

import pytest

from vla_zoo import load_model


def test_pi0_default_is_remote_first_without_local_load() -> None:
    model = load_model("pi0")

    assert model.name == "pi0"
    assert model.action_spec.shape == (32,)
    with pytest.raises(NotImplementedError, match="enable_local=True"):
        model.predict(image=None, instruction="test")
