"""Low-rate VLM success detector: publishes whether the current instruction is done.

Runs beside ``vla_runtime_node`` and never publishes actions. Each check asks the
VLM about the latest camera frame on a worker thread so a slow model cannot
block the executor.
"""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from time import monotonic
from typing import Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vla_ros2_msgs.msg import VLAInstruction as VLAInstructionMsg
from vla_ros2_msgs.msg import VLATaskStatus as VLATaskStatusMsg

from vla_ros2.vlm import (
    DEFAULT_PROMPT_TEMPLATE,
    SuccessDetector,
    SuccessJudgement,
    SuccessLatch,
    load_vlm,
)
from vla_ros2_ros.converters import ros_image_to_numpy
from vla_ros2_ros.qos import image_qos, instruction_qos, status_qos


class SuccessDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("vla_success_detector_node")
        self.declare_parameter("vlm_name", "dummy")
        self.declare_parameter("pretrained", "")
        self.declare_parameter("device", "auto")
        self.declare_parameter("dummy_answer", "no")
        self.declare_parameter("image_splitting", True)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("instruction_topic", "/vla/instruction")
        self.declare_parameter("instruction_msg_type", "string")
        self.declare_parameter("task_status_topic", "/vla/task_status")
        self.declare_parameter("check_hz", 0.5)
        self.declare_parameter("required_consecutive", 2)
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("use_logits", True)
        self.declare_parameter("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        self.declare_parameter("stale_image_timeout_sec", 2.0)

        def value(name: str) -> Any:
            return self.get_parameter(name).value

        vlm_name = str(value("vlm_name"))
        vlm_kwargs: dict[str, Any] = {
            "device": str(value("device")),
            "image_splitting": bool(value("image_splitting")),
        }
        if str(value("pretrained")):
            vlm_kwargs["pretrained"] = str(value("pretrained"))
        if vlm_name == "dummy":
            vlm_kwargs = {"answer": str(value("dummy_answer"))}
        self._vlm = load_vlm(vlm_name, **vlm_kwargs)
        self._detector = SuccessDetector(
            self._vlm,
            prompt_template=str(value("prompt_template")),
            threshold=float(value("threshold")),
            use_logits=bool(value("use_logits")),
        )
        self._latch = SuccessLatch(int(value("required_consecutive")))
        self._stale_image_timeout_sec = float(value("stale_image_timeout_sec"))

        self._latest_image: Image | None = None
        self._latest_image_received_at: float | None = None
        self._instruction = ""
        self._task_id = ""
        self._checks = 0
        self._pending: Future[SuccessJudgement] | None = None
        self._pending_instruction = ""
        self._executor = ThreadPoolExecutor(max_workers=1)

        self.create_subscription(Image, str(value("image_topic")), self._image_cb, image_qos())
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
        self.create_timer(1.0 / max(1e-3, float(value("check_hz"))), self._tick)
        self.get_logger().info(
            f"vla_success_detector_node ready: vlm={self._vlm.name} model={self._vlm.model_id}"
        )

    def _image_cb(self, msg: Image) -> None:
        self._latest_image = msg
        self._latest_image_received_at = monotonic()

    def _set_instruction(self, text: str, task_id: str) -> None:
        text = text.strip()
        if text == self._instruction and task_id == self._task_id:
            return
        self._instruction = text
        self._task_id = task_id
        self._latch.reset()
        self.get_logger().info(f"new instruction, success latch reset: {text!r}")

    def _image_is_fresh(self) -> bool:
        if self._latest_image_received_at is None:
            return False
        return monotonic() - self._latest_image_received_at <= self._stale_image_timeout_sec

    def _tick(self) -> None:
        if self._pending is not None and not self._pending.done():
            return
        if not self._instruction or self._latest_image is None or not self._image_is_fresh():
            return
        try:
            image = ros_image_to_numpy(self._latest_image)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        self._pending_instruction = self._instruction
        self._pending = self._executor.submit(self._detector.judge, self._instruction, [image])
        self._pending.add_done_callback(self._handle_judgement)

    def _handle_judgement(self, future: Future[SuccessJudgement]) -> None:
        try:
            judgement = future.result()
        except Exception as exc:
            self.get_logger().error(f"VLM success check failed: {exc}")
            return
        if self._pending_instruction != self._instruction:
            return  # instruction changed while the VLM was thinking; drop stale result
        self._checks += 1
        succeeded = self._latch.update(judgement)
        self._publish(judgement, succeeded)

    def _publish(self, judgement: SuccessJudgement, succeeded: bool) -> None:
        msg = VLATaskStatusMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.task_id = self._task_id
        msg.instruction = self._instruction
        msg.vlm_name = self._vlm.name
        msg.model_id = self._vlm.model_id
        msg.success = succeeded
        msg.last_check_success = bool(judgement.success)
        msg.confidence = float(judgement.confidence if judgement.confidence is not None else -1.0)
        msg.answer_text = judgement.answer
        msg.consecutive_success = self._latch.consecutive
        msg.checks = self._checks
        msg.latency_ms = float(judgement.latency_ms)
        msg.metadata_json = json.dumps(
            {**judgement.metadata, "required_consecutive": self._latch.required_consecutive}
        )
        self._status_pub.publish(msg)

    def destroy_node(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SuccessDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
