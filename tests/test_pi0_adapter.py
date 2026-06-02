from __future__ import annotations

import pytest

from vla_zoo import load_model
from vla_zoo.adapters.pi0 import _pi0_local_load_error, _pi0_tokenizer_repo


def test_pi0_default_is_remote_first_without_local_load() -> None:
    model = load_model("pi0")

    assert model.name == "pi0"
    assert model.action_spec.shape == (32,)
    with pytest.raises(NotImplementedError, match="enable_local=True"):
        model.predict(image=None, instruction="test")


def test_pi0_tokenizer_repo_extracted_from_preprocessor() -> None:
    preprocessor = {
        "steps": [
            {"registry_name": "to_batch_processor", "config": {}},
            {
                "registry_name": "tokenizer_processor",
                "config": {"tokenizer_name": "google/paligemma-3b-pt-224"},
            },
        ]
    }
    assert _pi0_tokenizer_repo(preprocessor) == "google/paligemma-3b-pt-224"


def test_pi0_tokenizer_repo_none_when_absent() -> None:
    assert _pi0_tokenizer_repo({"steps": [{"registry_name": "to_batch_processor"}]}) is None
    assert _pi0_tokenizer_repo({}) is None


def test_pi0_preflight_refuses_missing_weights() -> None:
    # Missing weights is the silent random-weight hazard: refuse loudly.
    message = _pi0_local_load_error(
        pretrained="lerobot/pi0_base",
        weights_status="missing",
        tokenizer_repo="google/paligemma-3b-pt-224",
        tokenizer_status="ok",
    )
    assert message is not None
    assert "randomly-initialized" in message
    assert "model.safetensors" in message


def test_pi0_preflight_points_at_gated_tokenizer_license() -> None:
    message = _pi0_local_load_error(
        pretrained="lerobot/pi0_base",
        weights_status="ok",
        tokenizer_repo="google/paligemma-3b-pt-224",
        tokenizer_status="gated",
    )
    assert message is not None
    assert "gated" in message
    assert "google/paligemma-3b-pt-224" in message
    assert "token" in message.lower()


def test_pi0_preflight_passes_when_assets_resolve() -> None:
    assert (
        _pi0_local_load_error(
            pretrained="lerobot/pi0_base",
            weights_status="ok",
            tokenizer_repo="google/paligemma-3b-pt-224",
            tokenizer_status="ok",
        )
        is None
    )
