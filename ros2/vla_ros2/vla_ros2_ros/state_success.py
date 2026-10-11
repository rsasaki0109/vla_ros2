"""Proprioceptive success detector: grasp -> release -> settle from joint_states.

Publishes ``VLATaskStatus`` like ``vla_success_detector_node`` but needs no camera
or VLM: it runs ``vla_ros2.task_success.GraspReleaseDetector`` on every joint
state and resets on each new instruction. It never publishes actions.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from vla_ros2_msgs.msg import VLAInstruction as VLAInstructionMsg
from vla_ros2_msgs.msg import VLATaskStatus as VLATaskStatusMsg

from vla_ros2.task_success import GraspReleaseConfig, GraspReleaseDetector
from vla_ros2_ros.qos import action_qos, instruction_qos, status_qos

# SO-100 order used by LeRobot: five arm joints then the gripper.
DEFAULT_STATE_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class StateSuccessNode(Node):
    def __init__(self) -> None:
        super().__init__("vla_state_success_node")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("instruction_topic", "/vla/instruction")
        self.declare_parameter("instruction_msg_type", "string")
        self.declare_parameter("task_status_topic", "/vla/task_status")
        self.declare_parameter("publish_hz", 2.0)
        # Joint names in GraspReleaseConfig index order (gripper_index/arm_indices).
        self.declare_parameter("state_joint_names", DEFAULT_STATE_JOINTS)
        # GraspReleaseConfig overrides as JSON, e.g. '{"hold_low": 0.01}' for metres.
        self.declare_parameter("config_json", "")

        def value(name: str) -> Any:
            return self.get_parameter(name).value

        overrides = json.loads(str(value("config_json")) or "{}")
        if "arm_indices" in overrides:
            overrides["arm_indices"] = tuple(overrides["arm_indices"])
        self._config = GraspReleaseConfig(**overrides)
        self._detector = GraspReleaseDetector(self._config)
        self._joint_names = [str(name) for name in value("state_joint_names")]
        self._instruction = ""
        self._task_id = ""
        self._samples = 0

        self.create_subscription(
            JointState, str(value("joint_state_topic")), self._joint_state_cb, action_qos(50)
        )
        instruction_topic = str(value("instruction_topic"))
        if str(value("instruction_msg_type")) == "vla_instruction":
            self.create_subscription(
                VLAInstructionMsg,
                instruction_topic,
                lambda msg: self._set_instruction(msg.text, msg.task_id),
                instruction_qos(10),
            )
        else:
            self.create_subscription(
                String,
                instruction_topic,
                lambda msg: self._set_instruction(msg.data, ""),
                instruction_qos(10),
            )
        self._status_pub = self.create_publisher(
            VLATaskStatusMsg, str(value("task_status_topic")), status_qos(10)
        )
        self.create_timer(1.0 / max(1e-3, float(value("publish_hz"))), self._publish)
        self.get_logger().info(
            f"vla_state_success_node ready: joints={self._joint_names} config={self._config}"
        )

    def _set_instruction(self, text: str, task_id: str) -> None:
        text = text.strip()
        if text == self._instruction and task_id == self._task_id:
            return
        self._instruction = text
        self._task_id = task_id
        self._detector.reset()
        self._samples = 0

    def _joint_state_cb(self, msg: JointState) -> None:
        if not self._instruction:
            return
        positions = dict(zip(msg.name, msg.position, strict=False))
        if not all(name in positions for name in self._joint_names):
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp == 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9
        self._detector.update([positions[name] for name in self._joint_names], stamp)
        self._samples += 1

    def _publish(self) -> None:
        if not self._instruction:
            return
        msg = VLATaskStatusMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.task_id = self._task_id
        msg.instruction = self._instruction
        msg.vlm_name = "grasp_release"
        msg.model_id = "vla_ros2.task_success.GraspReleaseDetector"
        msg.success = self._detector.success
        msg.last_check_success = self._detector.success
        msg.confidence = 1.0 if self._detector.success else 0.0
        msg.answer_text = self._detector.phase
        msg.checks = self._samples
        msg.metadata_json = json.dumps(
            {
                "phase": self._detector.phase,
                "hold_value": self._detector.hold_value,
                "config": asdict(self._config),
            }
        )
        self._status_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = StateSuccessNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
