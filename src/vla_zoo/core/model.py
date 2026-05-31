from __future__ import annotations

from typing import Any

from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation


class BaseVLA:
    """Base runtime boundary for Vision-Language-Action models."""

    name: str = "base"
    model_id: str = "base"
    action_spec: ActionSpec

    def __init__(
        self,
        *,
        name: str | None = None,
        model_id: str | None = None,
        action_spec: ActionSpec | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        if model_id is not None:
            self.model_id = model_id
        if action_spec is not None:
            self.action_spec = action_spec
        self.metadata: dict[str, Any] = dict(metadata or {})

    @classmethod
    def from_config(cls, **kwargs: Any) -> BaseVLA:
        """Construct an adapter from keyword configuration."""

        return cls(**kwargs)

    def predict(
        self,
        image: Any | None = None,
        instruction: str | None = None,
        *,
        observation: VLAObservation | None = None,
        **kwargs: Any,
    ) -> VLAAction | VLAActionChunk:
        """Predict an action from either a full observation or convenience inputs."""

        if observation is None:
            if instruction is None:
                msg = "instruction is required when observation is not provided"
                raise ValueError(msg)
            images = {} if image is None else {"primary": image}
            observation = VLAObservation(
                instruction=instruction,
                images=images,
                metadata=dict(kwargs),
            )
        elif kwargs:
            observation.metadata.update(kwargs)
        return self.predict_observation(observation)

    def predict_observation(self, observation: VLAObservation) -> VLAAction | VLAActionChunk:
        """Predict from the normalized observation contract."""

        raise NotImplementedError


class VLAAdapter(BaseVLA):
    """Marker class for concrete adapters."""
