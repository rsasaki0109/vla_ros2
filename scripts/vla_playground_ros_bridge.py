"""ROS2 bridge for the vla_ros2 browser playground (no real robot required).

Publishes ``/vla/instruction`` (and optional camera overrides), subscribes to
``/vla/action``, ``/vla/status``, and ``/camera/image_raw`` for live display.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class RosPlaygroundSnapshot:
    connected: bool
    error: str
    action_json: str
    status_json: str
    action_bars: dict[str, float] | None
    camera: Image.Image | None


def _require_ros() -> Any:
    try:
        import rclpy
    except ImportError as exc:
        msg = "ROS2 Python (rclpy) required; source /opt/ros/jazzy/setup.bash"
        raise RuntimeError(msg) from exc
    return rclpy


class RosPlaygroundBridge:
    """Background rclpy node for Gradio polling."""

    def __init__(
        self,
        *,
        instruction_topic: str = "/vla/instruction",
        action_topic: str = "/vla/action",
        status_topic: str = "/vla/status",
        image_topic: str = "/camera/image_raw",
    ) -> None:
        rclpy = _require_ros()
        from rclpy.node import Node
        from sensor_msgs.msg import Image as RosImage
        from vla_ros2_msgs.msg import VLAAction, VLAInstruction, VLAStatus
        from vla_ros2_ros.qos import action_qos, image_qos, instruction_qos, status_qos

        self._lock = threading.Lock()
        self._error = ""
        self._latest_action: VLAAction | None = None
        self._latest_status: VLAStatus | None = None
        self._latest_camera: Image.Image | None = None

        if not rclpy.ok():
            rclpy.init()

        class _BridgeNode(Node):
            def __init__(self, outer: RosPlaygroundBridge) -> None:
                super().__init__("vla_playground_bridge")
                self._outer = outer
                self._instruction_pub = self.create_publisher(
                    VLAInstruction,
                    instruction_topic,
                    instruction_qos(10),
                )
                self.create_subscription(
                    VLAAction,
                    action_topic,
                    self._action_cb,
                    action_qos(10),
                )
                self.create_subscription(
                    VLAStatus,
                    status_topic,
                    self._status_cb,
                    status_qos(10),
                )
                self.create_subscription(
                    RosImage,
                    image_topic,
                    self._image_cb,
                    image_qos(),
                )
                self.get_logger().info(
                    f"playground bridge on {instruction_topic} / {action_topic} / {status_topic}"
                )

            def _action_cb(self, msg: VLAAction) -> None:
                with self._outer._lock:
                    self._outer._latest_action = msg

            def _status_cb(self, msg: VLAStatus) -> None:
                with self._outer._lock:
                    self._outer._latest_status = msg

            def _image_cb(self, msg: RosImage) -> None:
                if msg.encoding not in {"rgb8", "bgr8"}:
                    return
                array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                if msg.encoding == "bgr8":
                    array = array[..., ::-1]
                with self._outer._lock:
                    self._outer._latest_camera = Image.fromarray(array.copy())

            def publish_instruction(self, text: str, *, task_id: str = "playground") -> None:
                msg = VLAInstruction()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "vla_playground"
                msg.text = text
                msg.task_id = task_id
                msg.metadata_json = json.dumps({"source": "vla_playground_ros"})
                self._instruction_pub.publish(msg)

        self._node = _BridgeNode(self)
        self._stop = False
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _spin_loop(self) -> None:
        rclpy = _require_ros()
        try:
            while not self._stop and rclpy.ok():
                rclpy.spin_once(self._node, timeout_sec=0.05)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._error = str(exc)

    def publish_instruction(self, text: str, *, repeat: int = 3) -> None:
        for _ in range(max(1, repeat)):
            self._node.publish_instruction(text)

    def snapshot(self) -> RosPlaygroundSnapshot:
        with self._lock:
            action = self._latest_action
            status = self._latest_status
            camera = self._latest_camera
            error = self._error

        action_json = "waiting for /vla/action..."
        action_bars: dict[str, float] | None = None
        if action is not None:
            payload = {
                "model_name": action.model_name,
                "adapter_name": action.adapter_name,
                "action_space": action.action_space,
                "data": list(action.data),
                "names": list(action.names),
                "metadata_json": action.metadata_json,
            }
            action_json = json.dumps(payload, indent=2)
            names = list(action.names) or [f"a{i}" for i in range(len(action.data))]
            action_bars = {
                str(name): float(value)
                for name, value in zip(names, action.data, strict=False)
            }

        status_json = "waiting for /vla/status..."
        if status is not None:
            status_json = json.dumps(
                {
                    "ready": status.ready,
                    "dry_run": status.dry_run,
                    "status_text": status.status_text,
                    "last_latency_ms": status.last_latency_ms,
                    "action_rate_hz": status.action_rate_hz,
                    "model_name": status.model_name,
                },
                indent=2,
            )

        return RosPlaygroundSnapshot(
            connected=error == "",
            error=error,
            action_json=action_json,
            status_json=status_json,
            action_bars=action_bars,
            camera=camera,
        )

    def close(self) -> None:
        self._stop = True
        self._spin_thread.join(timeout=2.0)
        self._node.destroy_node()

    @staticmethod
    def shutdown_ros() -> None:
        rclpy = _require_ros()
        if rclpy.ok():
            rclpy.shutdown()
