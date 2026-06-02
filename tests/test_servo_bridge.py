from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from vla_zoo.core.errors import ConfigurationError
from vla_zoo.core.types import ActionSpec, VLAAction
from vla_zoo.runtime.guard import ActionClipGuard
from vla_zoo.runtime.servo_bridge import (
    ServoJointJog,
    ServoTwist,
    action_to_servo_command,
    eef_delta_to_twist,
    joint_action_to_jog,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _eef(values: list[float]) -> VLAAction:
    spec = ActionSpec(
        action_space="eef_delta",
        shape=(len(values),),
        names=("x", "y", "z", "roll", "pitch", "yaw", "gripper")[: len(values)],
    )
    return VLAAction(data=np.asarray(values, dtype=np.float32), spec=spec)


def test_eef_delta_maps_to_twist_with_gripper() -> None:
    twist = eef_delta_to_twist(_eef([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]), frame_id="tool0")

    assert isinstance(twist, ServoTwist)
    assert twist.frame_id == "tool0"
    np.testing.assert_allclose(twist.linear, (0.1, 0.2, 0.3))
    np.testing.assert_allclose(twist.angular, (0.4, 0.5, 0.6))
    assert twist.gripper == 1.0


def test_eef_delta_scales_and_omits_absent_gripper() -> None:
    twist = eef_delta_to_twist(
        _eef([1.0, 0.0, 0.0, 0.0, 0.0, 2.0]), linear_scale=0.5, angular_scale=0.1
    )

    np.testing.assert_allclose(twist.linear, (0.5, 0.0, 0.0))
    np.testing.assert_allclose(twist.angular, (0.0, 0.0, 0.2))
    assert twist.gripper is None


def test_eef_delta_rejects_wrong_action_space() -> None:
    spec = ActionSpec(action_space="gripper", shape=(1,))
    action = VLAAction(data=np.zeros((1,), dtype=np.float32), spec=spec)
    with pytest.raises(ConfigurationError, match="eef_delta"):
        eef_delta_to_twist(action)


def test_joint_action_maps_to_jog() -> None:
    spec = ActionSpec(
        action_space="joint_velocity", shape=(2,), names=("joint_a", "joint_b")
    )
    action = VLAAction(data=np.asarray([0.2, -0.4], dtype=np.float32), spec=spec)
    jog = joint_action_to_jog(action, scale=2.0)

    assert isinstance(jog, ServoJointJog)
    assert jog.joint_names == ("joint_a", "joint_b")
    np.testing.assert_allclose(jog.velocities, (0.4, -0.8))


def test_dispatch_selects_command_type() -> None:
    assert isinstance(action_to_servo_command(_eef([0.0] * 7)), ServoTwist)
    spec = ActionSpec(action_space="joint_position", shape=(1,))
    joint = VLAAction(data=np.zeros((1,), dtype=np.float32), spec=spec)
    assert isinstance(action_to_servo_command(joint), ServoJointJog)


def test_dispatch_rejects_unsupported_space() -> None:
    spec = ActionSpec(action_space="base_twist", shape=(3,))
    action = VLAAction(data=np.zeros((3,), dtype=np.float32), spec=spec)
    with pytest.raises(ConfigurationError, match="No MoveIt Servo mapping"):
        action_to_servo_command(action)


def test_clip_then_map_pipeline_uses_guarded_action() -> None:
    spec = ActionSpec(
        action_space="eef_delta",
        shape=(6,),
        low=(-0.5,) * 6,
        high=(0.5,) * 6,
    )
    action = VLAAction(data=np.asarray([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), spec=spec)
    guard = ActionClipGuard()
    guarded = guard.clip(action)
    assert isinstance(guarded, VLAAction)
    twist = action_to_servo_command(guarded)

    # the out-of-bounds x is clamped to 0.5 before mapping
    assert isinstance(twist, ServoTwist)
    np.testing.assert_allclose(twist.linear, (0.5, 0.0, 0.0))
    assert guard.clipped_actions == 1


def test_moveit_servo_bridge_example_imports_without_rclpy() -> None:
    # The example must be importable for its pure dependencies; rclpy is imported lazily
    # inside main(), so importing the module must not require ROS2.
    spec = importlib.util.spec_from_file_location(
        "moveit_servo_bridge",
        REPO_ROOT / "examples" / "ros2" / "moveit_servo_bridge.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
    assert hasattr(module, "parse_args")
