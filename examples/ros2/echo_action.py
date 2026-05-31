from __future__ import annotations

import rclpy
from rclpy.node import Node
from vla_zoo_msgs.msg import VLAAction


class ActionEcho(Node):
    def __init__(self) -> None:
        super().__init__("vla_action_echo")
        self.create_subscription(VLAAction, "/vla/action", self.callback, 10)

    def callback(self, msg: VLAAction) -> None:
        self.get_logger().info(f"{msg.action_space}: {list(msg.data)}")


def main() -> None:
    rclpy.init()
    node = ActionEcho()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
