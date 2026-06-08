"""Reference controller bridge: parse /vla/action for Phase C bring-up (log-only by default).

This node is a starting point for robot-specific bridges. It does not claim to be
Cartesian IK or a production controller — it validates parsing and optional command
forwarding behind a second interlock (`enable_actuation`).
"""

from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String
from vla_ros2_msgs.msg import VLAAction

from vla_ros2_ros.action_parse import (
    ParsedVLAAction,
    parse_action_fields,
    parsed_twist_from_eef_delta,
)
from vla_ros2_ros.qos import action_qos


def _bool_param(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _twist_msg_from_parsed(parsed: ParsedVLAAction) -> Twist | None:
    twist = parsed_twist_from_eef_delta(parsed)
    if twist is None:
        return None
    msg = Twist()
    msg.linear.x = twist.linear_x
    msg.linear.y = twist.linear_y
    msg.linear.z = twist.linear_z
    msg.angular.x = twist.angular_x
    msg.angular.y = twist.angular_y
    msg.angular.z = twist.angular_z
    return msg


class VLAControllerBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("vla_controller_bridge_node")
        self.declare_parameter("action_topic", "/vla/action")
        self.declare_parameter("parsed_topic", "/vla/bridge/parsed")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("enable_actuation", False)
        self.declare_parameter("publish_parsed", True)
        self.declare_parameter("publish_cmd_vel", False)
        self.declare_parameter("log_every_n", 5)

        self._action_topic = str(self.get_parameter("action_topic").value)
        self._parsed_topic = str(self.get_parameter("parsed_topic").value)
        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._enable_actuation = _bool_param(self.get_parameter("enable_actuation").value)
        self._publish_parsed = _bool_param(self.get_parameter("publish_parsed").value)
        self._publish_cmd_vel = _bool_param(self.get_parameter("publish_cmd_vel").value)
        self._log_every_n = max(1, int(self.get_parameter("log_every_n").value))
        self._received = 0

        self._parsed_pub = (
            self.create_publisher(String, self._parsed_topic, 10) if self._publish_parsed else None
        )
        self._cmd_vel_pub = (
            self.create_publisher(Twist, self._cmd_vel_topic, 10)
            if self._publish_cmd_vel
            else None
        )
        self.create_subscription(
            VLAAction,
            self._action_topic,
            self._action_cb,
            action_qos(10),
        )
        self.get_logger().info(
            "vla_controller_bridge_node listening on "
            f"{self._action_topic}; actuation="
            f"{'enabled' if self._enable_actuation else 'disabled (log/parse only)'}"
        )

    def _action_cb(self, msg: VLAAction) -> None:
        parsed = parse_action_fields(
            model_name=msg.model_name,
            adapter_name=msg.adapter_name,
            action_space=msg.action_space,
            control_mode=msg.control_mode,
            frame_id=msg.frame_id,
            dt=msg.dt,
            data=list(msg.data),
            names=list(msg.names),
            metadata_json=msg.metadata_json,
        )
        self._received += 1
        if self._received == 1 or self._received % self._log_every_n == 0:
            summary = {
                "count": self._received,
                "action_space": parsed.action_space,
                "frame_id": parsed.frame_id,
                "named_values": parsed.named_values,
            }
            self.get_logger().info(f"parsed action: {json.dumps(summary, sort_keys=True)}")

        if self._parsed_pub is not None:
            out = String()
            payload = parsed.to_dict()
            payload["count"] = self._received
            out.data = json.dumps(payload, sort_keys=True)
            self._parsed_pub.publish(out)

        if not self._enable_actuation:
            return

        twist = _twist_msg_from_parsed(parsed)
        if twist is None:
            self.get_logger().warning(
                f"enable_actuation=true but action_space={parsed.action_space!r} "
                "is not supported by the reference Twist mapper"
            )
            return
        if self._cmd_vel_pub is not None:
            self._cmd_vel_pub.publish(twist)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VLAControllerBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
