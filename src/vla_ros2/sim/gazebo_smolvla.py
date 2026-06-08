"""Map between Gazebo 7-DoF arm joints and LeRobot SmolVLA SO-100 6D state/actions."""

from __future__ import annotations

import numpy as np

GZ_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)
GRIPPER_OPEN_M = 0.04


def _positions_by_name(names: list[str], positions: list[float]) -> dict[str, float]:
    return {
        name: float(position)
        for name, position in zip(names, positions, strict=False)
        if name
    }


def gazebo_joints_to_smolvla_state(names: list[str], positions: list[float]) -> list[float]:
    """Convert Gazebo ``joint_states`` (radians + gripper meters) to 6D SmolVLA proprio.

    SO-100 SmolVLA uses five arm joints in degrees plus a normalized gripper channel.
    ``joint_6`` on the Gazebo stand-in is not part of the 6D checkpoint vector.
    """
    by_name = _positions_by_name(names, positions)
    return [
        float(np.rad2deg(by_name.get("joint_1", 0.0))),
        float(np.rad2deg(by_name.get("joint_2", 0.0))),
        float(np.rad2deg(by_name.get("joint_3", 0.0))),
        float(np.rad2deg(by_name.get("joint_4", 0.0))),
        float(np.rad2deg(by_name.get("joint_5", 0.0))),
        float(by_name.get("gripper", 0.0) / GRIPPER_OPEN_M),
    ]


def smolvla_action_to_gazebo_positions(data: list[float]) -> dict[str, float]:
    """Convert a 6D SmolVLA command (5x degrees + gripper 0-1) to Gazebo joint targets."""
    if len(data) < 6:
        msg = f"expected 6D SmolVLA action, got {len(data)}"
        raise ValueError(msg)
    return {
        "joint_1": float(np.deg2rad(data[0])),
        "joint_2": float(np.deg2rad(data[1])),
        "joint_3": float(np.deg2rad(data[2])),
        "joint_4": float(np.deg2rad(data[3])),
        "joint_5": float(np.deg2rad(data[4])),
        "joint_6": 0.0,
        "gripper": float(np.clip(data[5], 0.0, 1.0) * GRIPPER_OPEN_M),
    }


def blend_joint_targets(
    *,
    current: dict[str, float],
    commanded: dict[str, float],
    blend: float,
    joint_limits: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Blend absolute SmolVLA targets toward the current Gazebo joint positions."""
    alpha = float(np.clip(blend, 0.0, 1.0))
    updated = dict(current)
    for name, target in commanded.items():
        if name not in updated:
            updated[name] = target
            continue
        value = updated[name] + (target - updated[name]) * alpha
        limits = joint_limits.get(name)
        if limits is not None:
            low, high = limits
            value = max(low, min(high, value))
        updated[name] = value
    return updated
