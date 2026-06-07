from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from time import monotonic

import rclpy
from builtin_interfaces.msg import Time
from rclpy._rclpy_pybind11 import RCLError
from rclpy.exceptions import ROSInterruptException
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vla_ros2_msgs.msg import VLAAction, VLAStatus

from vla_ros2_ros.qos import action_qos, status_qos


def _bool_param(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _stamp_to_sec(payload: dict[str, object]) -> float:
    header = payload.get("header")
    if not isinstance(header, dict):
        return 0.0
    stamp = header.get("stamp")
    if not isinstance(stamp, dict):
        return 0.0
    return float(stamp.get("sec", 0)) + float(stamp.get("nanosec", 0)) / 1_000_000_000.0


def _stamp_from_payload(payload: dict[str, object]) -> Time:
    msg = Time()
    header = payload.get("header")
    stamp = header.get("stamp") if isinstance(header, dict) else None
    if isinstance(stamp, dict):
        msg.sec = int(stamp.get("sec", 0))
        msg.nanosec = int(stamp.get("nanosec", 0))
    return msg


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    values: list[float] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int | float):
            values.append(float(item))
    return values


def _metadata(payload: dict[str, object]) -> dict[str, object]:
    raw = payload.get("metadata")
    if isinstance(raw, dict):
        return dict(raw)
    raw_text = payload.get("metadata_json")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {}
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"metadata_json": raw_text}
    return parsed if isinstance(parsed, dict) else {"metadata_json": raw_text}


def load_action_records(path: Path, *, max_actions: int = 0) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        if "data" not in payload:
            raise ValueError(f"{path}:{line_number} is missing action data")
        records.append(payload)
        if max_actions > 0 and len(records) >= max_actions:
            break
    return records


def action_record_to_msg(
    payload: dict[str, object],
    *,
    stamp: Time,
    frame_id_override: str,
    replay_index: int,
    source_path: str,
) -> VLAAction:
    msg = VLAAction()
    msg.header.stamp = stamp
    header = payload.get("header")
    frame_id = header.get("frame_id") if isinstance(header, dict) else ""
    msg.header.frame_id = frame_id_override or str(frame_id or "")
    msg.model_name = str(payload.get("model_name", ""))
    msg.adapter_name = str(payload.get("adapter_name", ""))
    msg.action_space = str(payload.get("action_space", "custom"))
    msg.control_mode = str(payload.get("control_mode", msg.action_space))
    msg.frame_id = frame_id_override or str(payload.get("frame_id", msg.header.frame_id))
    msg.dt = float(payload.get("dt", 0.0) or 0.0)
    msg.data = _float_list(payload.get("data"))
    msg.names = _string_list(payload.get("names"))
    confidence = payload.get("confidence", -1.0)
    msg.confidence = float(confidence) if isinstance(confidence, int | float) else -1.0
    chunk_index = payload.get("chunk_index", 0)
    msg.chunk_index = int(chunk_index) if isinstance(chunk_index, int) else 0
    metadata = {
        **_metadata(payload),
        "replay": True,
        "replay_index": replay_index,
        "replay_source": source_path,
        "original_stamp_sec": _stamp_to_sec(payload),
    }
    msg.metadata_json = json.dumps(metadata, separators=(",", ":"))
    return msg


class VLAActionReplayNode(Node):
    def __init__(self) -> None:
        super().__init__("vla_action_replay_node")
        self.declare_parameter("action_log_path", "results/ros2_smoke/vla_actions.jsonl")
        self.declare_parameter("action_topic", "/vla/action_replay")
        self.declare_parameter("status_topic", "/vla/replay_status")
        self.declare_parameter("use_recorded_timing", True)
        self.declare_parameter("replay_hz", 5.0)
        self.declare_parameter("speed", 1.0)
        self.declare_parameter("start_delay_sec", 1.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("stamp_now", True)
        self.declare_parameter("frame_id_override", "")
        self.declare_parameter("max_actions", 0)
        self.declare_parameter("max_queue_size", 10)

        self.action_log_path = Path(str(self.get_parameter("action_log_path").value))
        self.action_topic = str(self.get_parameter("action_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.use_recorded_timing = _bool_param(self.get_parameter("use_recorded_timing").value)
        self.replay_hz = float(self.get_parameter("replay_hz").value)
        self.speed = max(1e-6, float(self.get_parameter("speed").value))
        self.start_delay_sec = max(0.0, float(self.get_parameter("start_delay_sec").value))
        self.loop = _bool_param(self.get_parameter("loop").value)
        self.stamp_now = _bool_param(self.get_parameter("stamp_now").value)
        self.frame_id_override = str(self.get_parameter("frame_id_override").value)
        self.max_actions = int(self.get_parameter("max_actions").value)
        max_queue_size = int(self.get_parameter("max_queue_size").value)

        if self.replay_hz <= 0:
            raise ValueError("replay_hz must be positive")
        self.records = load_action_records(self.action_log_path, max_actions=self.max_actions)
        if not self.records:
            raise ValueError(f"no action records found in {self.action_log_path}")
        self.relative_times = self._relative_times(self.records)
        self.action_pub = self.create_publisher(
            VLAAction,
            self.action_topic,
            action_qos(max_queue_size),
        )
        self.status_pub = self.create_publisher(
            VLAStatus,
            self.status_topic,
            status_qos(max_queue_size),
        )

        self._start_time = monotonic() + self.start_delay_sec
        self._next_index = 0
        self._published = 0
        self._done = False
        self._tick_count = 0
        period = 0.02 if self.use_recorded_timing else 1.0 / self.replay_hz
        self.timer = self.create_timer(period, self._tick)
        self.status_timer = self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            "vla_action_replay_node ready: "
            f"records={len(self.records)} topic={self.action_topic} "
            f"start_delay_sec={self.start_delay_sec}"
        )

    def _relative_times(self, records: list[dict[str, object]]) -> list[float]:
        if not self.use_recorded_timing:
            return [index / self.replay_hz for index in range(len(records))]
        base = min(_stamp_to_sec(record) for record in records)
        return [max(0.0, _stamp_to_sec(record) - base) for record in records]

    def _reset(self) -> None:
        self._start_time = monotonic() + self.start_delay_sec
        self._next_index = 0
        self._done = False

    def _tick(self) -> None:
        self._tick_count += 1
        if self._tick_count == 1:
            self.get_logger().debug(
                "first replay tick: "
                f"use_recorded_timing={self.use_recorded_timing} replay_hz={self.replay_hz}"
            )
        if self._done:
            return
        now = monotonic()
        if now < self._start_time:
            return
        elapsed = (now - self._start_time) * self.speed
        published_this_tick = 0
        while self._next_index < len(self.records):
            if self.relative_times[self._next_index] > elapsed:
                break
            self._publish_action(self._next_index)
            self._next_index += 1
            self._published += 1
            published_this_tick += 1
            if not self.use_recorded_timing:
                break
        if self._next_index >= len(self.records):
            if self.loop:
                self._reset()
            else:
                self._done = True
        if published_this_tick:
            self._publish_status()

    def _publish_action(self, index: int) -> None:
        self.get_logger().debug(f"publishing replay action index={index}")
        stamp = (
            self.get_clock().now().to_msg()
            if self.stamp_now
            else _stamp_from_payload(self.records[index])
        )
        self.action_pub.publish(
            action_record_to_msg(
                self.records[index],
                stamp=stamp,
                frame_id_override=self.frame_id_override,
                replay_index=index,
                source_path=str(self.action_log_path),
            )
        )

    def _publish_status(self) -> None:
        msg = VLAStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.model_name = "action_replay"
        msg.adapter_name = "VLAActionReplayNode"
        msg.ready = not self._done
        msg.dry_run = True
        msg.last_latency_ms = 0.0
        msg.avg_latency_ms = 0.0
        msg.action_rate_hz = float(self.replay_hz)
        msg.dropped_frames = 0
        waiting = (not self._done) and monotonic() < self._start_time
        msg.status_text = "done" if self._done else "waiting" if waiting else "replaying"
        msg.metadata_json = json.dumps(
            {
                "action_log_path": str(self.action_log_path),
                "action_topic": self.action_topic,
                "records": len(self.records),
                "published": self._published,
                "next_index": self._next_index,
                "use_recorded_timing": self.use_recorded_timing,
                "speed": self.speed,
                "start_delay_sec": self.start_delay_sec,
                "loop": self.loop,
            },
            separators=(",", ":"),
        )
        self.status_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VLAActionReplayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, ROSInterruptException):
        pass
    except RCLError as exc:
        message = str(exc)
        if "context is not valid" not in message and "rcl_shutdown already called" not in message:
            raise
    finally:
        with suppress(RCLError):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
