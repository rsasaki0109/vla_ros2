"""Action helper functions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from vla_zoo.core.types import ActionSpec, VLAAction


def zeros_action(spec: ActionSpec, **metadata: object) -> VLAAction:
    """Create a zero-valued action for a spec."""

    return VLAAction(
        data=np.zeros(spec.shape, dtype=np.float32),
        spec=spec,
        metadata=dict(metadata),
    )


def clip_action(action: VLAAction) -> VLAAction:
    """Clip an action to its spec bounds when both low and high are declared."""

    if action.spec.low is None or action.spec.high is None:
        return action
    low = np.asarray(action.spec.low, dtype=np.float32).reshape(action.spec.shape)
    high = np.asarray(action.spec.high, dtype=np.float32).reshape(action.spec.shape)
    return VLAAction(
        data=np.clip(action.data, low, high),
        spec=action.spec,
        dt=action.dt,
        confidence=action.confidence,
        chunk_index=action.chunk_index,
        metadata={**action.metadata, "clipped": True},
    )


def action_names(spec: ActionSpec) -> Iterable[str]:
    """Return declared action names or stable positional names."""

    if spec.names:
        return spec.names
    return (f"action_{index}" for index in range(int(np.prod(spec.shape))))
