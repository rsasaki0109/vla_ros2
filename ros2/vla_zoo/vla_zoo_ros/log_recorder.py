from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

import rclpy
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from vla_zoo_msgs.msg import VLAStatus

from vla_zoo_ros.qos import status_qos


def _stamp_to_dict(stamp: Time) -> dict[str, int]:
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


def _key_value_to_dict(value: KeyValue) -> dict[str, str]:
    return {"key": value.key, "value": value.value}


def _diagnostic_status_to_dict(status: DiagnosticStatus) -> dict[str, Any]:
    return {
        "name": status.name,
        "hardware_id": status.hardware_id,
        "level": int(status.level),
        "message": status.message,
        "values": [_key_value_to_dict(value) for value in status.values],
    }


def _bool_param(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def status_msg_to_dict(msg: VLAStatus) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "header": {
            "stamp": _stamp_to_dict(msg.header.stamp),
            "frame_id": msg.header.frame_id,
        },
        "model_name": msg.model_name,
        "adapter_name": msg.adapter_name,
        "ready": bool(msg.ready),
        "dry_run": bool(msg.dry_run),
        "last_latency_ms": float(msg.last_latency_ms),
        "avg_latency_ms": float(msg.avg_latency_ms),
        "action_rate_hz": float(msg.action_rate_hz),
        "dropped_frames": int(msg.dropped_frames),
        "status_text": msg.status_text,
        "metadata_json": msg.metadata_json,
    }
    if msg.metadata_json:
        try:
            payload["metadata"] = json.loads(msg.metadata_json)
        except json.JSONDecodeError:
            payload["metadata"] = {}
    return payload


def diagnostics_msg_to_dict(msg: DiagnosticArray) -> dict[str, Any]:
    return {
        "header": {
            "stamp": _stamp_to_dict(msg.header.stamp),
            "frame_id": msg.header.frame_id,
        },
        "status": [_diagnostic_status_to_dict(status) for status in msg.status],
    }


class RuntimeLogRecorder(Node):
    def __init__(self) -> None:
        super().__init__("vla_runtime_log_recorder")
        self.declare_parameter("status_topic", "/vla/status")
        self.declare_parameter("diagnostics_topic", "/diagnostics")
        self.declare_parameter("status_log_path", "results/vla_status.jsonl")
        self.declare_parameter("diagnostics_log_path", "results/vla_diagnostics.jsonl")
        self.declare_parameter("record_status", True)
        self.declare_parameter("record_diagnostics", True)
        self.declare_parameter("max_records", 0)
        self.declare_parameter("flush_every", 1)

        self.status_topic = str(self.get_parameter("status_topic").value)
        self.diagnostics_topic = str(self.get_parameter("diagnostics_topic").value)
        self.record_status = _bool_param(self.get_parameter("record_status").value)
        self.record_diagnostics = _bool_param(self.get_parameter("record_diagnostics").value)
        self.max_records = int(self.get_parameter("max_records").value)
        self.flush_every = max(1, int(self.get_parameter("flush_every").value))
        self._records_written = 0
        self._status_file: TextIO | None = None
        self._diagnostics_file: TextIO | None = None

        if self.record_status:
            self._status_file = self._open_log(str(self.get_parameter("status_log_path").value))
            self.create_subscription(
                VLAStatus,
                self.status_topic,
                self._status_cb,
                status_qos(10),
            )
        if self.record_diagnostics:
            self._diagnostics_file = self._open_log(
                str(self.get_parameter("diagnostics_log_path").value)
            )
            self.create_subscription(
                DiagnosticArray,
                self.diagnostics_topic,
                self._diagnostics_cb,
                status_qos(10),
            )

        self.get_logger().info(
            "vla_runtime_log_recorder ready: "
            f"status={self.record_status} diagnostics={self.record_diagnostics}"
        )

    def _open_log(self, path_text: str) -> TextIO:
        path = Path(path_text).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8")

    def _write_jsonl(self, file: TextIO | None, payload: dict[str, Any]) -> None:
        if file is None:
            return
        if self.max_records > 0 and self._records_written >= self.max_records:
            return
        file.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._records_written += 1
        if self._records_written % self.flush_every == 0:
            file.flush()

    def _status_cb(self, msg: VLAStatus) -> None:
        self._write_jsonl(self._status_file, status_msg_to_dict(msg))

    def _diagnostics_cb(self, msg: DiagnosticArray) -> None:
        self._write_jsonl(self._diagnostics_file, diagnostics_msg_to_dict(msg))

    def destroy_node(self) -> None:
        for file in (self._status_file, self._diagnostics_file):
            if file is not None:
                file.flush()
                file.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RuntimeLogRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
