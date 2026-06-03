from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

import rclpy
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from vla_zoo_msgs.msg import VLAAction, VLAStatus

from vla_zoo.runtime.diagnostics import native_records_from_diagnostics_payload
from vla_zoo_ros.qos import action_qos, status_qos


def _stamp_to_dict(stamp: Time) -> dict[str, int]:
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


def _key_value_to_dict(value: KeyValue) -> dict[str, str]:
    return {"key": value.key, "value": value.value}


def _uint8_to_int(value: object) -> int:
    if isinstance(value, bytes):
        return int.from_bytes(value[:1], byteorder="little") if value else 0
    return int(value)  # type: ignore[arg-type]


def _diagnostic_status_to_dict(status: DiagnosticStatus) -> dict[str, Any]:
    return {
        "name": status.name,
        "hardware_id": status.hardware_id,
        "level": _uint8_to_int(status.level),
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


def action_msg_to_dict(msg: VLAAction) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "header": {
            "stamp": _stamp_to_dict(msg.header.stamp),
            "frame_id": msg.header.frame_id,
        },
        "model_name": msg.model_name,
        "adapter_name": msg.adapter_name,
        "action_space": msg.action_space,
        "control_mode": msg.control_mode,
        "frame_id": msg.frame_id,
        "dt": float(msg.dt),
        "data": [float(value) for value in msg.data],
        "names": list(msg.names),
        "confidence": float(msg.confidence),
        "chunk_index": int(msg.chunk_index),
        "metadata_json": msg.metadata_json,
    }
    if msg.metadata_json:
        try:
            payload["metadata"] = json.loads(msg.metadata_json)
        except json.JSONDecodeError:
            payload["metadata"] = {}
    return payload


class RuntimeLogRecorder(Node):
    def __init__(self) -> None:
        super().__init__("vla_runtime_log_recorder")
        self.declare_parameter("action_topic", "/vla/action")
        self.declare_parameter("status_topic", "/vla/status")
        self.declare_parameter("diagnostics_topic", "/diagnostics")
        self.declare_parameter("action_log_path", "results/vla_actions.jsonl")
        self.declare_parameter("status_log_path", "results/vla_status.jsonl")
        self.declare_parameter("diagnostics_log_path", "results/vla_diagnostics.jsonl")
        self.declare_parameter(
            "native_diagnostics_log_path", "results/vla_diagnostics_native.jsonl"
        )
        self.declare_parameter("record_actions", True)
        self.declare_parameter("record_status", True)
        self.declare_parameter("record_diagnostics", True)
        self.declare_parameter("record_native_diagnostics", True)
        self.declare_parameter("native_diagnostics_status_name", "")
        self.declare_parameter("max_records", 0)
        self.declare_parameter("flush_every", 1)

        self.action_topic = str(self.get_parameter("action_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.diagnostics_topic = str(self.get_parameter("diagnostics_topic").value)
        self.record_actions = _bool_param(self.get_parameter("record_actions").value)
        self.record_status = _bool_param(self.get_parameter("record_status").value)
        self.record_diagnostics = _bool_param(self.get_parameter("record_diagnostics").value)
        self.record_native_diagnostics = _bool_param(
            self.get_parameter("record_native_diagnostics").value
        )
        native_name = str(self.get_parameter("native_diagnostics_status_name").value)
        self.native_diagnostics_status_name = native_name or None
        self.max_records = int(self.get_parameter("max_records").value)
        self.flush_every = max(1, int(self.get_parameter("flush_every").value))
        self._records_written = 0
        self._native_written = 0
        self._action_file: TextIO | None = None
        self._status_file: TextIO | None = None
        self._diagnostics_file: TextIO | None = None
        self._native_diagnostics_file: TextIO | None = None

        if self.record_actions:
            self._action_file = self._open_log(str(self.get_parameter("action_log_path").value))
            self.create_subscription(
                VLAAction,
                self.action_topic,
                self._action_cb,
                action_qos(10),
            )
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
            if self.record_native_diagnostics:
                self._native_diagnostics_file = self._open_log(
                    str(self.get_parameter("native_diagnostics_log_path").value)
                )
            self.create_subscription(
                DiagnosticArray,
                self.diagnostics_topic,
                self._diagnostics_cb,
                status_qos(10),
            )

        self.get_logger().info(
            "vla_runtime_log_recorder ready: "
            f"actions={self.record_actions} status={self.record_status} "
            f"diagnostics={self.record_diagnostics}"
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

    def _action_cb(self, msg: VLAAction) -> None:
        self._write_jsonl(self._action_file, action_msg_to_dict(msg))

    def _status_cb(self, msg: VLAStatus) -> None:
        self._write_jsonl(self._status_file, status_msg_to_dict(msg))

    def _diagnostics_cb(self, msg: DiagnosticArray) -> None:
        payload = diagnostics_msg_to_dict(msg)
        self._write_jsonl(self._diagnostics_file, payload)
        if self._native_diagnostics_file is not None:
            records = native_records_from_diagnostics_payload(
                payload, status_name=self.native_diagnostics_status_name
            )
            for record in records:
                self._write_native(record.to_dict())

    def _write_native(self, payload: dict[str, Any]) -> None:
        file = self._native_diagnostics_file
        if file is None:
            return
        if self.max_records > 0 and self._native_written >= self.max_records:
            return
        file.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._native_written += 1
        if self._native_written % self.flush_every == 0:
            file.flush()

    def destroy_node(self) -> None:
        for file in (
            self._action_file,
            self._status_file,
            self._diagnostics_file,
            self._native_diagnostics_file,
        ):
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
