"""Observation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vla_zoo.core.types import VLAObservation


def make_observation(
    *,
    image: Any | None = None,
    instruction: str,
    state: Mapping[str, Any] | None = None,
    timestamp: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> VLAObservation:
    """Build a normalized observation from common single-image inputs."""

    images = {} if image is None else {"primary": image}
    return VLAObservation(
        instruction=instruction,
        images=images,
        state=dict(state or {}),
        timestamp=timestamp,
        metadata=dict(metadata or {}),
    )
