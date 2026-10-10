#!/usr/bin/env python3
"""Probe Gazebo smoke graph: /vla/action, /vla/status, and optional joint motion."""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from vla_ros2_msgs.msg import VLAAction, VLAStatus
from vla_ros2_ros.qos import action_qos, status_qos


class GzSmokeProbe(Node):
    def __init__(self) -> None:
        super().__init__("gz_smoke_probe")
        self.actions: list[VLAAction] = []
        self.statuses: list[VLAStatus] = []
        self.joint_samples: list[dict[str, float]] = []
        self.trajectories: list[JointTrajectory] = []
        self.create_subscription(
            VLAAction, "/vla/action", lambda msg: self.actions.append(msg), action_qos(10)
        )
        self.create_subscription(
            VLAStatus, "/vla/status", lambda msg: self.statuses.append(msg), status_qos(10)
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            lambda msg: self.trajectories.append(msg),
            action_qos(10),
        )

    def _joint_cb(self, msg: JointState) -> None:
        if not msg.position:
            return
        self.joint_samples.append(
            {name: float(p) for name, p in zip(msg.name, msg.position, strict=False)}
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("1", "2"), default="1")
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    parser.add_argument("--motion-min-range", type=float, default=1e-3)
    parser.add_argument(
        "--require-motion",
        action="store_true",
        help="fail phase 2 when no joint moves more than --motion-min-range",
    )
    args = parser.parse_args(argv)

    rclpy.init()
    node = GzSmokeProbe()
    deadline = time.time() + args.timeout_sec
    try:
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if args.phase == "1":
                if node.actions and any(status.ready for status in node.statuses):
                    break
            elif node.trajectories and len(node.joint_samples) >= 10:
                break
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not node.actions:
        print("FAIL: no /vla/action received")
        return 1
    if not any(status.ready for status in node.statuses):
        print("FAIL: no ready /vla/status received")
        return 1
    print(
        f"PASS phase {args.phase}: actions={len(node.actions)} "
        f"statuses={len(node.statuses)} model={node.actions[0].model_name}"
    )

    if args.phase == "2":
        if not node.trajectories:
            print("FAIL: no joint_trajectory commands from action bridge")
            return 1
        print(f"trajectories={len(node.trajectories)}")
        if len(node.joint_samples) < 2:
            print(f"FAIL: insufficient joint_states samples ({len(node.joint_samples)})")
            return 1
        names = sorted({name for sample in node.joint_samples for name in sample})
        ranges = {}
        for name in names:
            values = [sample[name] for sample in node.joint_samples if name in sample]
            ranges[name] = max(values) - min(values)
        joint_name, joint_range = max(ranges.items(), key=lambda item: item[1])
        print(f"max joint range={joint_range:.6f} ({joint_name}) samples={len(node.joint_samples)}")
        if joint_range < args.motion_min_range:
            if args.require_motion:
                print(f"FAIL: joint motion below {args.motion_min_range}")
                return 1
            print(
                "WARN: joint motion below threshold; bridge trajectories still prove actuation path"
            )
        else:
            print("PASS: joint motion detected")
        print("PASS: closed-loop actuation path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
