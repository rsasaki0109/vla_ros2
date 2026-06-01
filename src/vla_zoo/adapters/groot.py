from __future__ import annotations

from typing import Any

from vla_zoo.core.errors import MissingDependencyError
from vla_zoo.core.model import VLAAdapter
from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation

GROOT_ACTION_SPEC = ActionSpec(
    action_space="custom",
    shape=(1,),
    description=(
        "Experimental GR00T placeholder; concrete humanoid action specs must override this."
    ),
)


class GR00TAdapter(VLAAdapter):
    """Experimental lazy scaffold for Isaac GR00T-style model stacks."""

    name = "groot"
    model_id = "isaac-groot"
    action_spec = GROOT_ACTION_SPEC
    experimental = True
    domain = "humanoid/generalist"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name=self.name,
            model_id=self.model_id,
            action_spec=self.action_spec,
            metadata=kwargs,
        )

    @classmethod
    def from_config(cls, **kwargs: Any) -> GR00TAdapter:
        return cls(**kwargs)

    def predict_observation(self, observation: VLAObservation) -> VLAAction | VLAActionChunk:
        try:
            __import__("gr00t")
        except ImportError as exc:
            msg = (
                "Isaac GR00T dependencies are not installed. "
                "Install NVIDIA's GR00T stack in the serving environment before using this adapter."
            )
            raise MissingDependencyError(msg) from exc
        msg = "GR00T local inference integration is experimental and not implemented in the MVP."
        raise NotImplementedError(msg)
