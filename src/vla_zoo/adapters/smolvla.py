from __future__ import annotations

from typing import Any

from vla_zoo.core.model import VLAAdapter
from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation

SMOLVLA_ACTION_SPEC = ActionSpec(
    action_space="eef_delta",
    shape=(7,),
    description=(
        "Placeholder SmolVLA action spec. SmolVLA deployments typically use multi-camera "
        "images, robot state, and action chunks; concrete adapters should override this."
    ),
)


class SmolVLAAdapter(VLAAdapter):
    """Lazy placeholder for LeRobot SmolVLA policies."""

    name = "smolvla"
    model_id = "lerobot/smolvla"
    action_spec = SMOLVLA_ACTION_SPEC

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name=self.name,
            model_id=self.model_id,
            action_spec=self.action_spec,
            metadata=kwargs,
        )

    @classmethod
    def from_config(cls, **kwargs: Any) -> SmolVLAAdapter:
        return cls(**kwargs)

    def predict_observation(self, observation: VLAObservation) -> VLAAction | VLAActionChunk:
        msg = (
            "Local SmolVLA inference is not implemented in vla_zoo MVP. "
            "Install LeRobot in a dedicated environment or use a remote adapter."
        )
        raise NotImplementedError(msg)
