"""SmolVLA-oriented synthetic camera + instruction publisher for closed-loop sim bring-up.

Subscribes to live ``/joint_states`` (e.g. from Gazebo), renders a 256x256 SO-100-style
top view, and publishes it as the runtime camera together with a fixed stacking instruction.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from vla_ros2_msgs.msg import VLAInstruction

from vla_ros2.sim.gazebo_smolvla import gazebo_joints_to_smolvla_state
from vla_ros2.sim.so100_kinematic import observation_images, scene_from_dataset_state
from vla_ros2_ros.qos import action_qos, image_qos, instruction_qos


class VLASmolVLAInputNode(Node):
    def __init__(self) -> None:
        super().__init__("vla_smolvla_input_node")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("instruction_topic", "/vla/instruction")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("publish_hz", 5.0)
        self.declare_parameter("image_size", 256)
        self.declare_parameter("instruction", "stack the red block on the blue block")
        self.declare_parameter("task_id", "gz_smolvla_stack")
        self.declare_parameter("metadata_json", "")
        self.declare_parameter("publish_instruction", True)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.instruction_topic = str(self.get_parameter("instruction_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        publish_hz = float(self.get_parameter("publish_hz").value)
        self.image_size = int(self.get_parameter("image_size").value)
        self.instruction = str(self.get_parameter("instruction").value)
        self.task_id = str(self.get_parameter("task_id").value)
        self.metadata_json = self._metadata_json(str(self.get_parameter("metadata_json").value))
        self.publish_instruction = bool(self.get_parameter("publish_instruction").value)

        if publish_hz <= 0:
            raise ValueError("publish_hz must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

        self._latest_joint_state: JointState | None = None
        self.image_pub = self.create_publisher(Image, self.image_topic, image_qos())
        self.instruction_pub = self.create_publisher(
            VLAInstruction,
            self.instruction_topic,
            instruction_qos(10),
        )
        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_cb,
            action_qos(10),
        )
        self.timer = self.create_timer(1.0 / publish_hz, self._publish)
        self.get_logger().info(
            f"vla_smolvla_input_node rendering {self.image_size}x{self.image_size} "
            f"from {self.joint_state_topic} -> {self.image_topic}"
        )

    def _metadata_json(self, raw: str) -> str:
        if not raw:
            return json.dumps({"source": "vla_smolvla_input_node", "scene": "so100_render"})
        parsed: Any = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("metadata_json must decode to a JSON object")
        return json.dumps(parsed)

    def _joint_state_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _publish(self) -> None:
        if self._latest_joint_state is None:
            return
        stamp = self.get_clock().now().to_msg()
        image = self._render_image()
        self.image_pub.publish(self._numpy_to_image(image, stamp))
        if self.publish_instruction:
            self.instruction_pub.publish(self._make_instruction(stamp))

    def _render_image(self) -> np.ndarray:
        msg = self._latest_joint_state
        assert msg is not None
        state = np.asarray(
            gazebo_joints_to_smolvla_state(list(msg.name), list(msg.position)),
            dtype=np.float32,
        )
        scene = scene_from_dataset_state(state)
        images = observation_images(scene, size=self.image_size)
        return images["camera1"]

    def _numpy_to_image(self, array: np.ndarray, stamp: Any) -> Image:
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB image, got shape {array.shape}")
        height, width, _ = array.shape
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "smolvla_camera1"
        msg.height = height
        msg.width = width
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = width * 3
        msg.data = array.astype(np.uint8, copy=False).tobytes()
        return msg

    def _make_instruction(self, stamp: Any) -> VLAInstruction:
        msg = VLAInstruction()
        msg.header.stamp = stamp
        msg.header.frame_id = "smolvla_camera1"
        msg.text = self.instruction
        msg.task_id = self.task_id
        msg.metadata_json = self.metadata_json
        return msg


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VLASmolVLAInputNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
