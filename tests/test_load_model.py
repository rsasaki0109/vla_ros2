from __future__ import annotations

import builtins

import pytest

from vla_ros2 import load_model
from vla_ros2.core.errors import MissingDependencyError, UnknownModelError


def test_load_model_dummy() -> None:
    model = load_model("dummy")
    assert model.name == "dummy"


def test_unknown_model() -> None:
    with pytest.raises(UnknownModelError):
        load_model("not-a-model")


def test_missing_openvla_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"torch", "transformers"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(MissingDependencyError, match=r'pip install "vla_ros2\[openvla\]"'):
        load_model("openvla")
