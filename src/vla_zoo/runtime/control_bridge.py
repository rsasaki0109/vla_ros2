"""Pure ros2_control bridge mapping: VLA joint actions -> JointTrajectory points.

Many arms are driven through ros2_control controllers (e.g.
``joint_trajectory_controller``) rather than MoveIt Servo. This module turns a joint-space
:class:`VLAAction` into a framework-agnostic trajectory-point description so the mapping is
unit-tested without rclpy. The ROS2 example builds the real
``trajectory_msgs/JointTrajectory`` message from these dataclasses.

The core never actuates: these helpers only describe the command. A real deployment must
still run the clip + staleness guards (see :mod:`vla_zoo.runtime.guard`), validate joint
limits, and add an e-stop below the controller.
"""

from __future__ import annotations

from dataclasses import dataclass

from vla_zoo.core.errors import ConfigurationError
from vla_zoo.core.types import VLAAction


@dataclass(frozen=True)
class JointTrajectoryPoint:
    """One ros2_control trajectory point (``trajectory_msgs/JointTrajectoryPoint``)."""

    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    time_from_start_sec: float

    def to_dict(self) -> dict[str, object]:
        return {
            "joint_names": list(self.joint_names),
            "positions": list(self.positions),
            "velocities": list(self.velocities),
            "time_from_start_sec": self.time_from_start_sec,
        }


def _joint_names(action: VLAAction, count: int) -> tuple[str, ...]:
    if action.spec.names:
        return tuple(action.spec.names)
    return tuple(f"joint_{index}" for index in range(count))


def joint_action_to_trajectory_point(
    action: VLAAction,
    *,
    time_from_start_sec: float = 0.1,
    scale: float = 1.0,
) -> JointTrajectoryPoint:
    """Map a joint-space action to a single ros2_control trajectory point.

    A ``joint_position`` action fills ``positions`` (velocities left empty); a
    ``joint_velocity`` action fills ``velocities`` (positions left empty). The
    ``time_from_start_sec`` controls how fast the controller is asked to reach the point;
    pick it from the VLA control rate. ``scale`` throttles raw VLA outputs.
    """

    space = action.spec.action_space
    if space not in ("joint_position", "joint_velocity"):
        msg = (
            "joint_action_to_trajectory_point requires a joint_position/joint_velocity "
            f"action, got {space!r}"
        )
        raise ConfigurationError(msg)
    if time_from_start_sec <= 0:
        msg = f"time_from_start_sec must be positive, got {time_from_start_sec}"
        raise ConfigurationError(msg)

    values = tuple(value * scale for value in action.tolist())
    names = _joint_names(action, len(values))
    empty: tuple[float, ...] = ()
    if space == "joint_position":
        positions, velocities = values, empty
    else:
        positions, velocities = empty, values
    return JointTrajectoryPoint(
        joint_names=names,
        positions=positions,
        velocities=velocities,
        time_from_start_sec=time_from_start_sec,
    )
