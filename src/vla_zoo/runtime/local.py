from __future__ import annotations

from typing import Any

from vla_zoo.core.model import BaseVLA
from vla_zoo.core.registry import load_model
from vla_zoo.core.types import VLAAction, VLAActionChunk, VLAObservation


class LocalVLARuntime:
    """Thin owner for an in-process VLA adapter."""

    def __init__(self, model: BaseVLA) -> None:
        self.model = model

    @classmethod
    def from_model_name(cls, name: str, **kwargs: Any) -> LocalVLARuntime:
        return cls(load_model(name, **kwargs))

    def predict(self, observation: VLAObservation) -> VLAAction | VLAActionChunk:
        return self.model.predict(observation=observation)
