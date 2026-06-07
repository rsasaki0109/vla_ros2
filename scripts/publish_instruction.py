#!/usr/bin/env python3
"""Publish one instruction on /vla/instruction with vla_ros2-compatible QoS."""

from __future__ import annotations

import argparse
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from vla_ros2_ros.qos import instruction_qos


class InstructionPublisher(Node):
    def __init__(self, topic: str, text: str, repeat_hz: float) -> None:
        super().__init__("vla_instruction_publisher")
        self._text = text
        self._pub = self.create_publisher(String, topic, instruction_qos(10))
        if repeat_hz > 0:
            self.create_timer(1.0 / repeat_hz, self._publish)
        else:
            self._publish()

    def _publish(self) -> None:
        msg = String()
        msg.data = self._text
        self._pub.publish(msg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/vla/instruction")
    parser.add_argument("--text", default="pick up the cup")
    parser.add_argument(
        "--repeat-hz",
        type=float,
        default=0.0,
        help="Publish repeatedly at this rate (0 = once then exit after one spin).",
    )
    parser.add_argument(
        "--hold-sec",
        type=float,
        default=0.5,
        help="Seconds to spin after a single publish so the message is delivered.",
    )
    args = parser.parse_args(argv)

    rclpy.init()
    node = InstructionPublisher(args.topic, args.text, args.repeat_hz)
    try:
        if args.repeat_hz > 0:
            rclpy.spin(node)
        else:
            end = node.get_clock().now().nanoseconds + int(args.hold_sec * 1e9)
            while rclpy.ok() and node.get_clock().now().nanoseconds < end:
                rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
