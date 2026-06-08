"""Pure helpers for parsing vla_ros2_msgs/VLAAction into controller-facing fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedVLAAction:
    model_name: str
    adapter_name: str
    action_space: str
    control_mode: str
    frame_id: str
    dt: float
    data: tuple[float, ...]
    names: tuple[str, ...]
    named_values: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "action_space": self.action_space,
            "control_mode": self.control_mode,
            "frame_id": self.frame_id,
            "dt": self.dt,
            "data": list(self.data),
            "names": list(self.names),
            "named_values": dict(self.named_values),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def _parse_metadata(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"metadata_json": raw}
    return parsed if isinstance(parsed, dict) else {"metadata_json": raw}


def parse_action_fields(
    *,
    model_name: str,
    adapter_name: str,
    action_space: str,
    control_mode: str,
    frame_id: str,
    dt: float,
    data: list[float] | tuple[float, ...],
    names: list[str] | tuple[str, ...],
    metadata_json: str = "",
) -> ParsedVLAAction:
    values = tuple(float(item) for item in data)
    labels = tuple(str(item) for item in names)
    named: dict[str, float] = {}
    for index, value in enumerate(values):
        key = labels[index] if index < len(labels) else f"dim_{index}"
        named[key] = value
    return ParsedVLAAction(
        model_name=model_name,
        adapter_name=adapter_name,
        action_space=action_space,
        control_mode=control_mode,
        frame_id=frame_id,
        dt=float(dt),
        data=values,
        names=labels,
        named_values=named,
        metadata=_parse_metadata(metadata_json),
    )


def eef_delta_named_values(named_values: dict[str, float]) -> dict[str, float]:
    """Return canonical eef_delta keys present in the action."""
    keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    return {key: float(named_values[key]) for key in keys if key in named_values}


@dataclass(frozen=True)
class ParsedTwist:
    linear_x: float = 0.0
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_x: float = 0.0
    angular_y: float = 0.0
    angular_z: float = 0.0


def parsed_twist_from_eef_delta(parsed: ParsedVLAAction) -> ParsedTwist | None:
    if parsed.action_space and parsed.action_space != "eef_delta":
        return None
    delta = eef_delta_named_values(parsed.named_values)
    if not delta:
        return None
    return ParsedTwist(
        linear_x=delta.get("x", 0.0),
        linear_y=delta.get("y", 0.0),
        linear_z=delta.get("z", 0.0),
        angular_x=delta.get("roll", 0.0),
        angular_y=delta.get("pitch", 0.0),
        angular_z=delta.get("yaw", 0.0),
    )
