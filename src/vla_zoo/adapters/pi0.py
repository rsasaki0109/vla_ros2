from __future__ import annotations

from typing import Any

from vla_zoo.core.model import VLAAdapter
from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation

PI0_ACTION_SPEC = ActionSpec(
    action_space="eef_delta",
    shape=(7,),
    description="Placeholder pi0/openpi action spec; real adapters must override this.",
)


class Pi0Adapter(VLAAdapter):
    """Remote-first scaffold for the openpi/pi0 model family."""

    name = "pi0"
    model_id = "pi0"
    action_spec = PI0_ACTION_SPEC

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name=self.name,
            model_id=self.model_id,
            action_spec=self.action_spec,
            metadata=kwargs,
        )

    @classmethod
    def from_config(cls, **kwargs: Any) -> Pi0Adapter:
        return cls(**kwargs)

    def predict_observation(self, observation: VLAObservation) -> VLAAction | VLAActionChunk:
        msg = (
            "Local pi0/openpi inference is not implemented in vla_zoo MVP. "
            "Run pi0 in a dedicated server and load it with runtime='remote'."
        )
        raise NotImplementedError(msg)
