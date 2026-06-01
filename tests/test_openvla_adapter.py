from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vla_zoo.adapters import openvla


class _FakeTensor:
    def __init__(self, *, floating: bool = False) -> None:
        self.floating = floating
        self.dtype: object | None = None

    def is_floating_point(self) -> bool:
        return self.floating

    def to(self, *, dtype: object | None = None) -> _FakeTensor:
        self.dtype = dtype
        return self


class _FakeInputs(dict[str, _FakeTensor]):
    device: str | None = None

    def to(self, device: str) -> _FakeInputs:
        self.device = device
        return self


class _FakeProcessor:
    @classmethod
    def from_pretrained(cls, pretrained: str, **kwargs: Any) -> _FakeProcessor:
        return cls()

    def __call__(self, prompt: str, image: object, return_tensors: str) -> _FakeInputs:
        return _FakeInputs(
            {
                "input_ids": _FakeTensor(floating=False),
                "attention_mask": _FakeTensor(floating=False),
                "pixel_values": _FakeTensor(floating=True),
            }
        )


class _FakeModel:
    last_predict_kwargs: dict[str, Any] | None = None

    def to(self, device: str) -> _FakeModel:
        return self

    def eval(self) -> None:
        return None

    def predict_action(self, **kwargs: Any) -> np.ndarray:
        type(self).last_predict_kwargs = kwargs
        return np.zeros(7, dtype=np.float32)


class _FakeAutoModel:
    last_kwargs: dict[str, Any] | None = None
    last_model: _FakeModel | None = None

    @classmethod
    def from_pretrained(cls, pretrained: str, **kwargs: Any) -> _FakeModel:
        cls.last_kwargs = kwargs
        cls.last_model = _FakeModel()
        return cls.last_model


class _FakeTorch:
    bfloat16 = object()


class _FakeImage:
    pass


class _FakeImageModule:
    Image = _FakeImage

    @staticmethod
    def fromarray(array: object) -> _FakeImage:
        return _FakeImage()


def test_openvla_adapter_uses_compatible_loader_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        openvla,
        "_import_openvla_dependencies",
        lambda: (_FakeTorch, _FakeImageModule, (_FakeProcessor, _FakeAutoModel)),
    )

    adapter = openvla.OpenVLAAdapter(pretrained="fake/openvla", device="cuda:0")
    action = adapter.predict(image=_FakeImage(), instruction="pick up the red block")

    assert action.data.shape == (7,)
    assert _FakeAutoModel.last_kwargs is not None
    assert _FakeAutoModel.last_kwargs["attn_implementation"] == "eager"
    assert _FakeAutoModel.last_kwargs["torch_dtype"] is _FakeTorch.bfloat16
    assert _FakeModel.last_predict_kwargs is not None
    assert "attention_mask" not in _FakeModel.last_predict_kwargs
    assert _FakeModel.last_predict_kwargs["pixel_values"].dtype is _FakeTorch.bfloat16
