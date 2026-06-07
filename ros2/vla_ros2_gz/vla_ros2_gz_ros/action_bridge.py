"""Bridge vla_ros2_msgs/VLAAction to joint_trajectory_controller commands.

Sim-oriented mapping: each element of a 7-DoF ``eef_delta`` action increments the
matching joint target by ``data[i] * joint_scales[i]``. This is a lightweight stand-in
for Cartesian IK on the bundled 7-DoF Gazebo arm; real robots should use a dedicated
controller bridge (see ros2/BRINGUP.md).
"""

from __future__ import annotations

import json
from typing import Any

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from vla_ros2_msgs.msg import VLAAction

_ACTION_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


def _parse_float_list(raw: object) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple)):
        return [float(item) for item in raw]
    return [float(raw)]


def _parse_string_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    return [str(raw)]


def integrate_joint_targets(
    *,
    data: list[float],
    joint_names: list[str],
    joint_scales: list[float],
    targets: dict[str, float],
    joint_limits: dict[str, tuple[float, float]],
) -> dict[str, float]:
    updated = dict(targets)
    for index, name in enumerate(joint_names):
        if index >= len(data):
            break
        scale = joint_scales[index] if index < len(joint_scales) else 0.0
        value = updated.get(name, 0.0) + float(data[index]) * scale
        limits = joint_limits.get(name)
        if limits is not None:
            low, high = limits
            value = max(low, min(high, value))
        updated[name] = value
    return updated


class VLAActionBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("vla_action_bridge_node")
        self.declare_parameter("action_topic", "/vla/action")
        self.declare_parameter(
            "trajectory_topic", "/joint_trajectory_controller/joint_trajectory"
        )
        self.declare_parameter(
            "joint_names",
            ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
        )
        self.declare_parameter("joint_scales", "0.15,0.15,0.15,0.35,0.35,0.35,0.01")
        self.declare_parameter("trajectory_time_sec", 0.2)
        self.declare_parameter("enable_actuation", False)
        self.declare_parameter("expected_action_space", "eef_delta")

        self._action_topic = str(self.get_parameter("action_topic").value)
        self._trajectory_topic = str(self.get_parameter("trajectory_topic").value)
        self._joint_names = _parse_string_list(self.get_parameter("joint_names").value)
        self._joint_scales = _parse_float_list(self.get_parameter("joint_scales").value)
        self._trajectory_time_sec = float(self.get_parameter("trajectory_time_sec").value)
        self._enable_actuation = bool(self.get_parameter("enable_actuation").value)
        self._expected_action_space = str(self.get_parameter("expected_action_space").value)

        self._targets: dict[str, float] = {name: 0.0 for name in self._joint_names}
        self._joint_limits: dict[str, tuple[float, float]] = {
            "joint_1": (-2.5, 2.5),
            "joint_2": (-2.5, 2.5),
            "joint_3": (-2.5, 2.5),
            "joint_4": (-2.5, 2.5),
            "joint_5": (-2.5, 2.5),
            "joint_6": (-2.5, 2.5),
            "gripper": (0.0, 0.04),
        }
        self._initialized_from_joint_state = False

        self._traj_pub = self.create_publisher(
            JointTrajectory,
            self._trajectory_topic,
            _ACTION_QOS,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            _ACTION_QOS,
        )
        self.create_subscription(
            VLAAction,
            self._action_topic,
            self._action_cb,
            _ACTION_QOS,
        )
        self.get_logger().info(
            f"vla_action_bridge_node listening on {self._action_topic}; "
            f"actuation={'enabled' if self._enable_actuation else 'disabled'}"
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        if self._initialized_from_joint_state:
            return
        for name, position in zip(msg.name, msg.position, strict=False):
            if name in self._targets:
                self._targets[name] = float(position)
        if any(name in msg.name for name in self._joint_names):
            self._initialized_from_joint_state = True

    def _action_cb(self, msg: VLAAction) -> None:
        if not self._enable_actuation:
            return
        if msg.action_space and msg.action_space != self._expected_action_space:
            self.get_logger().warning(
                f"unsupported action_space {msg.action_space!r}; "
                f"expected {self._expected_action_space!r}"
            )
            return
        data = [float(item) for item in msg.data]
        self._targets = integrate_joint_targets(
            data=data,
            joint_names=self._joint_names,
            joint_scales=self._joint_scales,
            targets=self._targets,
            joint_limits=self._joint_limits,
        )
        self._publish_trajectory(msg)

    def _publish_trajectory(self, action: VLAAction) -> None:
        point = JointTrajectoryPoint()
        point.positions = [self._targets[name] for name in self._joint_names]
        duration_ns = int(max(0.05, self._trajectory_time_sec) * 1_000_000_000)
        point.time_from_start = Duration(
            sec=duration_ns // 1_000_000_000,
            nanosec=duration_ns % 1_000_000_000,
        )

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.header.frame_id = action.frame_id or "base_link"
        traj.joint_names = list(self._joint_names)
        traj.points = [point]
        metadata = {
            "source": "vla_action_bridge_node",
            "model_name": action.model_name,
            "adapter_name": action.adapter_name,
            "action_space": action.action_space,
        }
        if action.metadata_json:
            try:
                parsed: Any = json.loads(action.metadata_json)
                if isinstance(parsed, dict):
                    metadata["action_metadata"] = parsed
            except json.JSONDecodeError:
                metadata["action_metadata_json"] = action.metadata_json
        self._traj_pub.publish(traj)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VLAActionBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
