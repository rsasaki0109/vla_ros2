"""MoveIt Servo action-bridge example (dry-run safe).

This example subscribes to ``/vla/action`` (vla_zoo_msgs/VLAAction), runs the pure
clip + staleness guards, maps the action onto a MoveIt Servo command, and (only when
``--engage`` is passed) republishes it to Servo. By default it is **dry-run safe**: it
logs the command it would send without publishing, so it is harmless to run against a
live Servo node.

It is an example, not part of the core package: the core publishes actions only and never
actuates. rclpy / ROS message imports are done lazily in ``main`` so this file documents
the contract and stays importable for the pure mapping it relies on.

Run (dry-run, prints the mapped command):

    python examples/ros2/moveit_servo_bridge.py

Engage publishing to MoveIt Servo (only on a configured, safe robot):

    python examples/ros2/moveit_servo_bridge.py --engage \
      --twist-topic /servo_node/delta_twist_cmds --frame-id base_link
"""

from __future__ import annotations

import argparse

from vla_zoo.runtime.guard import ActionClipGuard
from vla_zoo.runtime.servo_bridge import (
    ServoJointJog,
    ServoTwist,
    action_to_servo_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoveIt Servo action bridge (dry-run safe).")
    parser.add_argument("--action-topic", default="/vla/action")
    parser.add_argument("--twist-topic", default="/servo_node/delta_twist_cmds")
    parser.add_argument("--joint-topic", default="/servo_node/delta_joint_cmds")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--linear-scale", type=float, default=1.0)
    parser.add_argument("--angular-scale", type=float, default=1.0)
    parser.add_argument("--joint-scale", type=float, default=1.0)
    parser.add_argument(
        "--engage",
        action="store_true",
        help="Publish to MoveIt Servo. Without this flag the bridge only logs (dry-run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy imports so the example documents the contract without requiring rclpy/MoveIt
    # to merely read or unit-test the pure mapping in vla_zoo.runtime.servo_bridge.
    import rclpy
    from control_msgs.msg import JointJog
    from geometry_msgs.msg import TwistStamped
    from rclpy.node import Node
    from vla_zoo_msgs.msg import VLAAction as VLAActionMsg

    from vla_zoo.core.types import ActionSpec, VLAAction

    class ServoBridge(Node):
        def __init__(self) -> None:
            super().__init__("vla_moveit_servo_bridge")
            self._clip = ActionClipGuard()
            self.create_subscription(VLAActionMsg, args.action_topic, self._on_action, 10)
            self.twist_pub = self.create_publisher(TwistStamped, args.twist_topic, 10)
            self.joint_pub = self.create_publisher(JointJog, args.joint_topic, 10)
            mode = "ENGAGED" if args.engage else "dry-run (logging only)"
            self.get_logger().info(f"MoveIt Servo bridge started in {mode} mode.")

        def _on_action(self, msg: VLAActionMsg) -> None:
            spec = ActionSpec(action_space=msg.action_space, shape=(len(msg.data),))
            action = VLAAction(data=list(msg.data), spec=spec)
            guarded = self._clip.clip(action)
            if not isinstance(guarded, VLAAction):
                guarded = guarded.actions[0]  # take the first action of a chunk
            command = action_to_servo_command(
                guarded,
                frame_id=args.frame_id,
                linear_scale=args.linear_scale,
                angular_scale=args.angular_scale,
                joint_scale=args.joint_scale,
            )
            if not args.engage:
                self.get_logger().info(f"[dry-run] would send: {command.to_dict()}")
                return
            self._publish(command)

        def _publish(self, command: ServoTwist | ServoJointJog) -> None:
            stamp = self.get_clock().now().to_msg()
            if isinstance(command, ServoTwist):
                twist = TwistStamped()
                twist.header.stamp = stamp
                twist.header.frame_id = command.frame_id
                twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = command.linear
                twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z = (
                    command.angular
                )
                self.twist_pub.publish(twist)
                return
            jog = JointJog()
            jog.header.stamp = stamp
            jog.joint_names = list(command.joint_names)
            jog.velocities = list(command.velocities)
            self.joint_pub.publish(jog)

    rclpy.init()
    node = ServoBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
