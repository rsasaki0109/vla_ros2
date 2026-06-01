from __future__ import annotations

from vla_zoo import list_models
from vla_zoo.core.registry import get_adapter_info


def test_list_models_contains_dummy() -> None:
    assert "dummy" in {model.name for model in list_models()}


def test_aliases_resolve() -> None:
    assert get_adapter_info("openpi").name == "pi0"
    assert get_adapter_info("gr00t").name == "groot"
