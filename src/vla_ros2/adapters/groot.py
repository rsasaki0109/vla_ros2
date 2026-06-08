from __future__ import annotations

from typing import Any

from vla_ros2.core.errors import MissingDependencyError
from vla_ros2.core.model import VLAAdapter
from vla_ros2.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation

GROOT_ACTION_SPEC = ActionSpec(
    action_space="custom",
    shape=(1,),
    description=(
        "Experimental GR00T placeholder; concrete humanoid action specs must override this."
    ),
)

#: A single, reusable note explaining why this adapter does not run yet. Used by the
#: adapter, the registry verification text, and the evidence matrix so the "blocked"
#: status reads identically everywhere.
GROOT_BLOCKED_NOTE = (
    "GR00T is blocked until the NVIDIA Isaac GR00T stack is wired in. This repository "
    "ships no GR00T inference and makes no task-success claim. Stand up the upstream "
    "stack in a dedicated serving environment and add a real serving adapter before use."
)


class GR00TAdapter(VLAAdapter):
    """Experimental lazy scaffold for Isaac GR00T-style model stacks.

    This adapter is intentionally inert: it declares the runtime contract but performs
    no inference, because a real serving integration requires the external NVIDIA Isaac
    GR00T stack. It must never fabricate actions to look functional.

    Expected observation/action contract (to be confirmed against the upstream stack):

    - Observation: multimodal RGB camera frame(s), an instruction/task string, and a
      humanoid proprioceptive state vector. Concrete camera count, image size, and state
      layout are stack/embodiment specific and are not pinned here.
    - Action: a humanoid/generalist action interface. GR00T-class models typically emit
      action chunks rather than a single step, so robot-side code should expect a
      ``VLAActionChunk``; the concrete shape and control rate are checkpoint specific and
      ``GROOT_ACTION_SPEC`` is a placeholder that a real adapter must override.
    """

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
            raise MissingDependencyError(GROOT_BLOCKED_NOTE) from exc
        # Even when the upstream package imports, this repository deliberately does not
        # fabricate inference: a real serving adapter is still required.
        raise NotImplementedError(GROOT_BLOCKED_NOTE)
