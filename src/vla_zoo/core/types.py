from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

ActionSpace = Literal[
    "eef_delta",
    "eef_pose",
    "joint_position",
    "joint_velocity",
    "base_twist",
    "gripper",
    "custom",
]

VALID_ACTION_SPACES: frozenset[str] = frozenset(
    {
        "eef_delta",
        "eef_pose",
        "joint_position",
        "joint_velocity",
        "base_twist",
        "gripper",
        "custom",
    }
)


@dataclass(frozen=True)
class ActionSpec:
    action_space: ActionSpace
    shape: tuple[int, ...]
    names: tuple[str, ...] = ()
    frame_id: str | None = None
    control_hz: float | None = None
    normalized: bool = False
    low: tuple[float, ...] | None = None
    high: tuple[float, ...] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.action_space not in VALID_ACTION_SPACES:
            msg = f"Unsupported action_space {self.action_space!r}"
            raise ValueError(msg)
        if not self.shape or any(dim <= 0 for dim in self.shape):
            msg = "shape must contain one or more positive dimensions"
            raise ValueError(msg)

        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        object.__setattr__(self, "names", tuple(self.names))
        if self.low is not None:
            object.__setattr__(self, "low", tuple(float(value) for value in self.low))
        if self.high is not None:
            object.__setattr__(self, "high", tuple(float(value) for value in self.high))

        size = prod(self.shape)
        if self.names and len(self.names) != size:
            msg = f"names length {len(self.names)} must match flattened action size {size}"
            raise ValueError(msg)
        if self.low is not None and len(self.low) != size:
            msg = f"low length {len(self.low)} must match flattened action size {size}"
            raise ValueError(msg)
        if self.high is not None and len(self.high) != size:
            msg = f"high length {len(self.high)} must match flattened action size {size}"
            raise ValueError(msg)
        if self.low is not None and self.high is not None:
            for low, high in zip(self.low, self.high, strict=True):
                if low > high:
                    msg = "low values must be <= high values"
                    raise ValueError(msg)
        if self.control_hz is not None and self.control_hz <= 0:
            msg = "control_hz must be positive when provided"
            raise ValueError(msg)


@dataclass
class VLAObservation:
    instruction: str
    images: Mapping[str, Any] = field(default_factory=dict)
    state: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VLAAction:
    data: NDArray[np.float32]
    spec: ActionSpec
    dt: float | None = None
    confidence: float | None = None
    chunk_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.data, dtype=np.float32)
        if array.shape != self.spec.shape:
            msg = f"action data shape {array.shape} does not match spec shape {self.spec.shape}"
            raise ValueError(msg)
        self.data = array

    def to_numpy(self) -> NDArray[np.float32]:
        """Return the action data as a NumPy array."""

        return np.asarray(self.data, dtype=np.float32)

    def tolist(self) -> list[float]:
        """Return the flattened action data as JSON-compatible floats."""

        return [float(value) for value in self.data.reshape(-1).tolist()]


@dataclass
class VLAActionChunk:
    actions: Sequence[VLAAction]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.actions:
            msg = "VLAActionChunk requires at least one action"
            raise ValueError(msg)

    def to_numpy(self) -> NDArray[np.float32]:
        """Stack actions into a NumPy array with shape (chunk, *action_shape)."""

        return np.stack([action.to_numpy() for action in self.actions]).astype(np.float32)
