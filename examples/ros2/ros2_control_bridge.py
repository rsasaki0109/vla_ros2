"""ros2_control action-bridge example (dry-run safe).

This example subscribes to ``/vla/action`` (vla_zoo_msgs/VLAAction), runs the pure
clip + staleness guards, maps a joint-space action onto a ``trajectory_msgs/JointTrajectory``
point, and (only when ``--engage`` is passed) republishes it to a ros2_control
``joint_trajectory_controller``. By default it is **dry-run safe**: it logs the trajectory
it would send without publishing.

It is an example, not part of the core package: the core publishes actions only and never
actuates. rclpy / ROS message imports are done lazily in ``main`` so this file documents
the contract and stays importable for the pure mapping it relies on.

Run (dry-run, prints the mapped trajectory point):

    python examples/ros2/ros2_control_bridge.py

Engage publishing on a configured, safe robot only:

    python examples/ros2/ros2_control_bridge.py --engage \
      --command-topic /joint_trajectory_controller/joint_trajectory \
      --time-from-start 0.1
"""

from __future__ import annotations

import argparse

from vla_zoo.runtime.control_bridge import joint_action_to_trajectory_point
from vla_zoo.runtime.guard import ActionClipGuard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ros2_control action bridge (dry-run safe).")
    parser.add_argument("--action-topic", default="/vla/action")
    parser.add_argument(
        "--command-topic", default="/joint_trajectory_controller/joint_trajectory"
    )
    parser.add_argument("--time-from-start", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--engage",
        action="store_true",
        help="Publish to the controller. Without this flag the bridge only logs (dry-run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy imports so the example documents the contract without requiring rclpy to merely
    # read or unit-test the pure mapping in vla_zoo.runtime.control_bridge.
    import rclpy
    from builtin_interfaces.msg import Duration
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from vla_zoo_msgs.msg import VLAAction as VLAActionMsg

    from vla_zoo.core.types import ActionSpec, VLAAction

    class ControlBridge(Node):
        def __init__(self) -> None:
            super().__init__("vla_ros2_control_bridge")
            self._clip = ActionClipGuard()
            self.create_subscription(VLAActionMsg, args.action_topic, self._on_action, 10)
            self.command_pub = self.create_publisher(JointTrajectory, args.command_topic, 10)
            mode = "ENGAGED" if args.engage else "dry-run (logging only)"
            self.get_logger().info(f"ros2_control bridge started in {mode} mode.")

        def _on_action(self, msg: VLAActionMsg) -> None:
            names = tuple(msg.names) if msg.names else ()
            spec = ActionSpec(
                action_space=msg.action_space, shape=(len(msg.data),), names=names
            )
            action = VLAAction(data=list(msg.data), spec=spec)
            guarded = self._clip.clip(action)
            if not isinstance(guarded, VLAAction):
                guarded = guarded.actions[0]  # take the first action of a chunk
            point = joint_action_to_trajectory_point(
                guarded, time_from_start_sec=args.time_from_start, scale=args.scale
            )
            if not args.engage:
                self.get_logger().info(f"[dry-run] would send: {point.to_dict()}")
                return

            traj = JointTrajectory()
            traj.header.stamp = self.get_clock().now().to_msg()
            traj.joint_names = list(point.joint_names)
            ros_point = JointTrajectoryPoint()
            ros_point.positions = list(point.positions)
            ros_point.velocities = list(point.velocities)
            seconds = int(point.time_from_start_sec)
            ros_point.time_from_start = Duration(
                sec=seconds,
                nanosec=int((point.time_from_start_sec - seconds) * 1e9),
            )
            traj.points = [ros_point]
            self.command_pub.publish(traj)

    rclpy.init()
    node = ControlBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
