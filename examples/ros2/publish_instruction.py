from __future__ import annotations

import argparse
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_zoo_msgs.msg import VLAInstruction


class InstructionPublisher(Node):
    def __init__(self, *, typed: bool, instruction: str, task_id: str) -> None:
        super().__init__("vla_instruction_publisher")
        self.typed = typed
        self.instruction = instruction
        self.task_id = task_id
        msg_type = VLAInstruction if typed else String
        self.publisher = self.create_publisher(msg_type, "/vla/instruction", 10)
        self.timer = self.create_timer(1.0, self.publish_once)

    def publish_once(self) -> None:
        if self.typed:
            msg = VLAInstruction()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.text = self.instruction
            msg.task_id = self.task_id
            msg.metadata_json = json.dumps({"source": "examples/ros2/publish_instruction.py"})
            self.publisher.publish(msg)
            self.get_logger().info(f"Published typed instruction: {msg.task_id} {msg.text}")
            return
        msg = String()
        msg.data = self.instruction
        self.publisher.publish(msg)
        self.get_logger().info(f"Published instruction: {msg.data}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a VLA instruction topic.")
    parser.add_argument("--typed", action="store_true", help="Publish vla_zoo_msgs/VLAInstruction.")
    parser.add_argument("--instruction", default="pick up the red block")
    parser.add_argument("--task-id", default="example_pick_red_block")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = InstructionPublisher(
        typed=args.typed,
        instruction=args.instruction,
        task_id=args.task_id,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
