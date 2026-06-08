from __future__ import annotations

from typing import Any

import numpy as np

from vla_ros2.core.model import VLAAdapter
from vla_ros2.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation

DEFAULT_DUMMY_SPEC = ActionSpec(
    action_space="eef_delta",
    shape=(7,),
    names=("x", "y", "z", "roll", "pitch", "yaw", "gripper"),
    frame_id="base_link",
    control_hz=5.0,
    normalized=True,
    description="Zero 7-DoF end-effector delta action for dry-run execution.",
)


class DummyAdapter(VLAAdapter):
    """Always-available adapter that returns neutral actions."""

    name = "dummy"
    model_id = "dummy"
    action_spec = DEFAULT_DUMMY_SPEC

    def __init__(
        self,
        *,
        action_spec: ActionSpec | None = None,
        chunk_size: int = 1,
        dt: float | None = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=self.name,
            model_id=self.model_id,
            action_spec=action_spec,
            metadata=kwargs,
        )
        if action_spec is not None:
            self.action_spec = action_spec
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            msg = "chunk_size must be positive"
            raise ValueError(msg)
        self.dt = dt

    @classmethod
    def from_config(cls, **kwargs: Any) -> DummyAdapter:
        return cls(**kwargs)

    def predict_observation(self, observation: VLAObservation) -> VLAAction | VLAActionChunk:
        metadata = {
            "model": self.name,
            "adapter": type(self).__name__,
            "instruction": observation.instruction,
            "dry_run": True,
        }
        if self.chunk_size == 1:
            return VLAAction(
                data=np.zeros(self.action_spec.shape, dtype=np.float32),
                spec=self.action_spec,
                dt=self.dt,
                confidence=1.0,
                metadata=metadata,
            )
        actions = [
            VLAAction(
                data=np.zeros(self.action_spec.shape, dtype=np.float32),
                spec=self.action_spec,
                dt=self.dt,
                confidence=1.0,
                chunk_index=index,
                metadata=metadata,
            )
            for index in range(self.chunk_size)
        ]
        return VLAActionChunk(actions=actions, metadata={**metadata, "chunk_size": self.chunk_size})
