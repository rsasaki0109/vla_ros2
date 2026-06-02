"""Pure MoveIt Servo bridge mapping: VLA actions -> Servo twist / joint-jog commands.

MoveIt Servo consumes ``geometry_msgs/TwistStamped`` (Cartesian jog) or
``control_msgs/JointJog`` (joint jog). This module turns a :class:`VLAAction` into a
framework-agnostic command description so the mapping is unit-tested without rclpy/MoveIt.
The ROS2 example builds the real messages from these dataclasses.

The core never actuates: these helpers only describe the command. A real deployment must
still run the clip + staleness guards (see :mod:`vla_zoo.runtime.guard`), keep MoveIt
Servo in dry-run/teleop-safe limits, and add an e-stop.
"""

from __future__ import annotations

from dataclasses import dataclass

from vla_zoo.core.errors import ConfigurationError
from vla_zoo.core.types import VLAAction

#: Canonical end-effector delta layout used by the built-in eef_delta adapters.
EEF_DELTA_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


@dataclass(frozen=True)
class ServoTwist:
    """A Cartesian jog command for MoveIt Servo (``geometry_msgs/TwistStamped``)."""

    frame_id: str
    linear: tuple[float, float, float]
    angular: tuple[float, float, float]
    gripper: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "twist",
            "frame_id": self.frame_id,
            "linear": list(self.linear),
            "angular": list(self.angular),
            "gripper": self.gripper,
        }


@dataclass(frozen=True)
class ServoJointJog:
    """A joint jog command for MoveIt Servo (``control_msgs/JointJog``)."""

    joint_names: tuple[str, ...]
    velocities: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "joint_jog",
            "joint_names": list(self.joint_names),
            "velocities": list(self.velocities),
        }


def eef_delta_to_twist(
    action: VLAAction,
    *,
    frame_id: str = "base_link",
    linear_scale: float = 1.0,
    angular_scale: float = 1.0,
) -> ServoTwist:
    """Map an ``eef_delta`` action to a Servo twist command.

    Expects ``[x, y, z, roll, pitch, yaw]`` and an optional 7th ``gripper`` element. The
    scales let the example throttle VLA outputs into Servo's safe jog range.
    """

    if action.spec.action_space != "eef_delta":
        msg = f"eef_delta_to_twist requires an eef_delta action, got {action.spec.action_space!r}"
        raise ConfigurationError(msg)
    values = action.tolist()
    if len(values) < 6:
        msg = f"eef_delta action must have at least 6 elements, got {len(values)}"
        raise ConfigurationError(msg)
    linear = (values[0] * linear_scale, values[1] * linear_scale, values[2] * linear_scale)
    angular = (values[3] * angular_scale, values[4] * angular_scale, values[5] * angular_scale)
    gripper = values[6] if len(values) >= 7 else None
    return ServoTwist(frame_id=frame_id, linear=linear, angular=angular, gripper=gripper)


def joint_action_to_jog(
    action: VLAAction,
    *,
    scale: float = 1.0,
) -> ServoJointJog:
    """Map a joint-space action (``joint_velocity``/``joint_position``) to a joint jog."""

    if action.spec.action_space not in ("joint_velocity", "joint_position"):
        msg = (
            "joint_action_to_jog requires a joint_velocity/joint_position action, "
            f"got {action.spec.action_space!r}"
        )
        raise ConfigurationError(msg)
    values = action.tolist()
    names = action.spec.names or tuple(f"joint_{index}" for index in range(len(values)))
    velocities = tuple(value * scale for value in values)
    return ServoJointJog(joint_names=tuple(names), velocities=velocities)


def action_to_servo_command(
    action: VLAAction,
    *,
    frame_id: str = "base_link",
    linear_scale: float = 1.0,
    angular_scale: float = 1.0,
    joint_scale: float = 1.0,
) -> ServoTwist | ServoJointJog:
    """Dispatch a VLA action to the matching MoveIt Servo command type."""

    space = action.spec.action_space
    if space == "eef_delta":
        return eef_delta_to_twist(
            action, frame_id=frame_id, linear_scale=linear_scale, angular_scale=angular_scale
        )
    if space in ("joint_velocity", "joint_position"):
        return joint_action_to_jog(action, scale=joint_scale)
    msg = (
        f"No MoveIt Servo mapping for action space {space!r}; supported: eef_delta, "
        "joint_velocity, joint_position."
    )
    raise ConfigurationError(msg)
