from __future__ import annotations

import json
from typing import Any

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from vla_ros2_msgs.msg import VLAAction as VLAActionMsg
from vla_ros2_msgs.msg import VLAActionChunk as VLAActionChunkMsg
from vla_ros2_msgs.msg import VLAStatus as VLAStatusMsg

from vla_ros2.core.types import VLAAction, VLAActionChunk


def ros_image_to_numpy(msg: Image) -> np.ndarray:
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }
    channels = channels_by_encoding.get(msg.encoding)
    if channels is None:
        msg_text = f"Unsupported image encoding {msg.encoding!r}; install cv_bridge if needed"
        raise ValueError(msg_text)
    array = np.frombuffer(msg.data, dtype=np.uint8)
    if channels == 1:
        return array.reshape((msg.height, msg.width))
    image = array.reshape((msg.height, msg.width, channels))
    if msg.encoding in {"bgr8", "bgra8"}:
        image = image[..., [2, 1, 0, *range(3, channels)]]
    return image


def _header(stamp: Time, frame_id: str | None = None) -> Header:
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id or ""
    return header


def action_to_msg(
    action: VLAAction,
    *,
    stamp: Time,
    model_name: str,
    adapter_name: str,
) -> VLAActionMsg:
    msg = VLAActionMsg()
    msg.header = _header(stamp, action.spec.frame_id)
    msg.model_name = model_name
    msg.adapter_name = adapter_name
    msg.action_space = action.spec.action_space
    msg.control_mode = action.spec.action_space
    msg.frame_id = action.spec.frame_id or ""
    msg.dt = float(action.dt or 0.0)
    msg.data = [float(value) for value in action.data.reshape(-1).tolist()]
    msg.names = list(action.spec.names)
    msg.confidence = float(action.confidence if action.confidence is not None else -1.0)
    msg.chunk_index = int(action.chunk_index if action.chunk_index is not None else 0)
    msg.metadata_json = json.dumps(action.metadata)
    return msg


def chunk_to_msg(
    chunk: VLAActionChunk,
    *,
    stamp: Time,
    model_name: str,
    adapter_name: str,
) -> VLAActionChunkMsg:
    msg = VLAActionChunkMsg()
    first = chunk.actions[0]
    msg.header = _header(stamp, first.spec.frame_id)
    msg.model_name = model_name
    msg.adapter_name = adapter_name
    msg.action_space = first.spec.action_space
    msg.action_dt = float(first.dt or 0.0)
    msg.actions = [
        action_to_msg(action, stamp=stamp, model_name=model_name, adapter_name=adapter_name)
        for action in chunk.actions
    ]
    msg.metadata_json = json.dumps(chunk.metadata)
    return msg


def status_to_msg(
    *,
    stamp: Time,
    model_name: str,
    adapter_name: str,
    ready: bool,
    dry_run: bool,
    last_latency_ms: float,
    avg_latency_ms: float,
    action_rate_hz: float,
    dropped_frames: int,
    status_text: str,
    metadata: dict[str, Any] | None = None,
) -> VLAStatusMsg:
    msg = VLAStatusMsg()
    msg.header = _header(stamp)
    msg.model_name = model_name
    msg.adapter_name = adapter_name
    msg.ready = ready
    msg.dry_run = dry_run
    msg.last_latency_ms = float(last_latency_ms)
    msg.avg_latency_ms = float(avg_latency_ms)
    msg.action_rate_hz = float(action_rate_hz)
    msg.dropped_frames = int(dropped_frames)
    msg.status_text = status_text
    msg.metadata_json = json.dumps(metadata or {})
    return msg
