from __future__ import annotations

from typing import Any

import numpy as np

from vla_ros2.adapters.dummy import DEFAULT_DUMMY_SPEC
from vla_ros2.core.model import VLAAdapter
from vla_ros2.core.types import ActionSpec, VLAAction, VLAObservation


def _phase_action(phase: str) -> tuple[float, float, float, float]:
    normalized = phase.strip().lower()
    if "approach" in normalized:
        return (0.8, -0.2, -0.3, 1.0)
    if "descend" in normalized or "place" in normalized:
        return (0.0, 0.0, -0.8, -0.6)
    if "close" in normalized:
        return (0.0, 0.0, 0.0, -1.0)
    if "lift" in normalized:
        return (0.0, 0.0, 0.8, -1.0)
    if "transport" in normalized:
        return (0.0, 0.9, 0.0, -1.0)
    if "open" in normalized:
        return (0.0, 0.0, 0.0, 1.0)
    if "retreat" in normalized:
        return (-0.7, -0.4, 0.6, 1.0)
    return (0.2, -0.1, 0.2, 1.0)


class RandomAdapter(VLAAdapter):
    """Deterministic random-action baseline for smoke comparisons."""

    name = "random"
    model_id = "random-baseline"
    action_spec = DEFAULT_DUMMY_SPEC

    def __init__(
        self,
        *,
        action_spec: ActionSpec | None = None,
        seed: int = 0,
        scale: float = 0.25,
        dt: float | None = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=self.name,
            model_id=self.model_id,
            action_spec=action_spec or self.action_spec,
            metadata=kwargs,
        )
        self.rng = np.random.default_rng(int(seed))
        self.scale = float(scale)
        self.dt = dt

    @classmethod
    def from_config(cls, **kwargs: Any) -> RandomAdapter:
        return cls(**kwargs)

    def predict_observation(self, observation: VLAObservation) -> VLAAction:
        data = self.rng.uniform(
            low=-self.scale,
            high=self.scale,
            size=self.action_spec.shape,
        ).astype(np.float32)
        return VLAAction(
            data=data,
            spec=self.action_spec,
            dt=self.dt,
            confidence=0.0,
            metadata={
                "model": self.name,
                "adapter": type(self).__name__,
                "baseline": "random",
                "instruction": observation.instruction,
                "scale": self.scale,
            },
        )


class ScriptedAdapter(VLAAdapter):
    """Rule-based baseline that emits phase-aware 7-DoF end-effector deltas."""

    name = "scripted"
    model_id = "scripted-baseline"
    action_spec = DEFAULT_DUMMY_SPEC

    def __init__(
        self,
        *,
        action_spec: ActionSpec | None = None,
        gain: float = 1.0,
        dt: float | None = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=self.name,
            model_id=self.model_id,
            action_spec=action_spec or self.action_spec,
            metadata=kwargs,
        )
        self.gain = float(gain)
        self.dt = dt

    @classmethod
    def from_config(cls, **kwargs: Any) -> ScriptedAdapter:
        return cls(**kwargs)

    def predict_observation(self, observation: VLAObservation) -> VLAAction:
        phase = str(observation.metadata.get("phase", ""))
        x, y, z, gripper = _phase_action(phase)
        data = np.zeros(self.action_spec.shape, dtype=np.float32)
        data[0] = np.clip(x * self.gain, -1.0, 1.0)
        data[1] = np.clip(y * self.gain, -1.0, 1.0)
        data[2] = np.clip(z * self.gain, -1.0, 1.0)
        data[6] = np.clip(gripper, -1.0, 1.0)
        return VLAAction(
            data=data,
            spec=self.action_spec,
            dt=self.dt,
            confidence=0.3,
            metadata={
                "model": self.name,
                "adapter": type(self).__name__,
                "baseline": "scripted",
                "phase": phase,
                "instruction": observation.instruction,
            },
        )
