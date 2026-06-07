from __future__ import annotations

import json
from typing import Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from vla_ros2_msgs.msg import VLAInstruction

from vla_ros2_ros.qos import action_qos, image_qos, instruction_qos


class VLASmokeInputNode(Node):
    def __init__(self) -> None:
        super().__init__("vla_smoke_input_node")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("instruction_topic", "/vla/instruction")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("publish_joint_state", True)
        self.declare_parameter("publish_hz", 5.0)
        self.declare_parameter("width", 320)
        self.declare_parameter("height", 240)
        self.declare_parameter("frame_id", "vla_smoke_camera")
        self.declare_parameter("instruction", "pick up the red block")
        self.declare_parameter("task_id", "ros2_smoke_pick_red_block")
        self.declare_parameter("metadata_json", "")

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.instruction_topic = str(self.get_parameter("instruction_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.publish_joint_state = bool(self.get_parameter("publish_joint_state").value)
        publish_hz = float(self.get_parameter("publish_hz").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.instruction = str(self.get_parameter("instruction").value)
        self.task_id = str(self.get_parameter("task_id").value)
        self.metadata_json = self._metadata_json(str(self.get_parameter("metadata_json").value))

        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if publish_hz <= 0:
            raise ValueError("publish_hz must be positive")

        self.image_pub = self.create_publisher(Image, self.image_topic, image_qos())
        self.instruction_pub = self.create_publisher(
            VLAInstruction,
            self.instruction_topic,
            instruction_qos(10),
        )
        self.joint_pub = self.create_publisher(JointState, self.joint_state_topic, action_qos(10))
        self.frame_index = 0
        self.timer = self.create_timer(1.0 / publish_hz, self._publish)
        self.get_logger().info(
            f"vla_smoke_input_node publishing image={self.image_topic} "
            f"instruction={self.instruction_topic} joint_state={self.publish_joint_state}"
        )

    def _metadata_json(self, raw: str) -> str:
        if not raw:
            return json.dumps(
                {
                    "source": "vla_smoke_input_node",
                    "scene": "synthetic_red_block",
                }
            )
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"metadata_json must be valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("metadata_json must decode to a JSON object")
        return json.dumps(parsed)

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(self._make_image(stamp))
        self.instruction_pub.publish(self._make_instruction(stamp))
        if self.publish_joint_state:
            self.joint_pub.publish(self._make_joint_state(stamp))
        self.frame_index += 1

    def _make_image(self, stamp: Any) -> Image:
        data = bytearray(self.width * self.height * 3)
        block_size = max(18, min(self.width, self.height) // 5)
        travel = max(1, self.width - block_size - 40)
        block_x = 20 + (self.frame_index * 3) % travel
        block_y = max(0, self.height // 2 - block_size // 2)
        table_y = max(0, int(self.height * 0.64))

        for y in range(self.height):
            for x in range(self.width):
                offset = (y * self.width + x) * 3
                if y >= table_y:
                    data[offset : offset + 3] = bytes((95, 108, 118))
                elif block_x <= x < block_x + block_size and block_y <= y < block_y + block_size:
                    data[offset : offset + 3] = bytes((220, 32, 32))
                else:
                    shade = 205 + ((x + y) % 24)
                    data[offset : offset + 3] = bytes((shade, shade, min(255, shade + 8)))

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = self.width * 3
        msg.data = bytes(data)
        return msg

    def _make_instruction(self, stamp: Any) -> VLAInstruction:
        msg = VLAInstruction()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.text = self.instruction
        msg.task_id = self.task_id
        msg.metadata_json = self.metadata_json
        return msg

    def _make_joint_state(self, stamp: Any) -> JointState:
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper",
        ]
        msg.position = [0.0, -0.35, 0.55, -0.2, 0.0, 0.25, 0.04]
        msg.velocity = [0.0] * len(msg.name)
        return msg


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VLASmokeInputNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
