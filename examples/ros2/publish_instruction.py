from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class InstructionPublisher(Node):
    def __init__(self) -> None:
        super().__init__("vla_instruction_publisher")
        self.publisher = self.create_publisher(String, "/vla/instruction", 10)
        self.timer = self.create_timer(1.0, self.publish_once)

    def publish_once(self) -> None:
        msg = String()
        msg.data = "pick up the red block"
        self.publisher.publish(msg)
        self.get_logger().info(f"Published instruction: {msg.data}")


def main() -> None:
    rclpy.init()
    node = InstructionPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
