from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from vla_zoo.core.errors import ConfigurationError
from vla_zoo.core.types import ActionSpec, VLAAction
from vla_zoo.runtime.control_bridge import (
    JointTrajectoryPoint,
    joint_action_to_trajectory_point,
)
from vla_zoo.runtime.guard import ActionClipGuard

REPO_ROOT = Path(__file__).resolve().parents[1]


def _joint(space: str, values: list[float], *, names=None, low=None, high=None) -> VLAAction:
    spec = ActionSpec(
        action_space=space,
        shape=(len(values),),
        names=tuple(names) if names else (),
        low=low,
        high=high,
    )
    return VLAAction(data=np.asarray(values, dtype=np.float32), spec=spec)


def test_joint_position_fills_positions_only() -> None:
    point = joint_action_to_trajectory_point(
        _joint("joint_position", [0.1, 0.2], names=("a", "b")),
        time_from_start_sec=0.2,
    )

    assert isinstance(point, JointTrajectoryPoint)
    assert point.joint_names == ("a", "b")
    np.testing.assert_allclose(point.positions, (0.1, 0.2))
    assert point.velocities == ()
    assert point.time_from_start_sec == 0.2


def test_joint_velocity_fills_velocities_only_with_scale() -> None:
    point = joint_action_to_trajectory_point(
        _joint("joint_velocity", [1.0, -2.0]),
        scale=0.5,
    )

    assert point.positions == ()
    np.testing.assert_allclose(point.velocities, (0.5, -1.0))
    # default joint names when the spec declares none
    assert point.joint_names == ("joint_0", "joint_1")


def test_rejects_non_joint_action_space() -> None:
    with pytest.raises(ConfigurationError, match="joint_position/joint_velocity"):
        joint_action_to_trajectory_point(_joint("eef_delta", [0.0] * 6))


def test_rejects_non_positive_time_from_start() -> None:
    with pytest.raises(ConfigurationError, match="time_from_start_sec"):
        joint_action_to_trajectory_point(
            _joint("joint_position", [0.0]), time_from_start_sec=0.0
        )


def test_clip_then_map_pipeline_clamps_before_trajectory() -> None:
    action = _joint(
        "joint_velocity",
        [5.0, 0.0],
        names=("a", "b"),
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
    )
    guard = ActionClipGuard()
    guarded = guard.clip(action)
    assert isinstance(guarded, VLAAction)
    point = joint_action_to_trajectory_point(guarded)

    np.testing.assert_allclose(point.velocities, (1.0, 0.0))
    assert guard.clipped_actions == 1


def test_ros2_control_bridge_example_imports_without_rclpy() -> None:
    spec = importlib.util.spec_from_file_location(
        "ros2_control_bridge",
        REPO_ROOT / "examples" / "ros2" / "ros2_control_bridge.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
    assert hasattr(module, "parse_args")
