"""ROS bag replay loader stub for benchmark credibility.

This is a deliberately scoped stub. It replays the JSONL action logs that vla_zoo
itself records (``vla_actions.jsonl``) and turns them into benchmark
:class:`~vla_zoo.benchmark.results.EpisodeRecord` rows plus a latency / action-rate
summary. It does **not** yet decode native rosbag2 archives (``.db3`` / ``.mcap``),
which require the ROS2 ``rosbag2`` stack; that path is intentionally left as future
work so this module stays importable without ROS2 installed.

A replay makes **no task-success claim**: ``success`` is recorded as ``None`` because
a recorded action stream does not, by itself, prove a task was completed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from vla_zoo.benchmark.results import EpisodeRecord

#: A single, reusable note describing exactly what this stub does and does not do.
ROSBAG_REPLAY_NOTE = (
    "ROS bag replay stub: replays recorded vla_zoo JSONL action logs (vla_actions.jsonl) "
    "for latency/action-rate analysis. Native rosbag2 (.db3/.mcap) decoding is not yet "
    "implemented and is gated on the ROS2 stack. No task-success claim is made."
)

REPLAY_SOURCE = "ros2-action-replay"


@dataclass(frozen=True)
class ReplayFrame:
    """One recorded action frame parsed from a JSONL action log."""

    index: int
    stamp_sec: float
    model_name: str
    action_space: str
    action: tuple[float, ...]
    names: tuple[str, ...]
    latency_ms: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "stamp_sec": self.stamp_sec,
            "model_name": self.model_name,
            "action_space": self.action_space,
            "action": list(self.action),
            "names": list(self.names),
            "latency_ms": self.latency_ms,
        }


def _stamp_seconds(entry: dict[str, object]) -> float:
    header = entry.get("header") or {}
    stamp = header.get("stamp") if isinstance(header, dict) else None
    if not isinstance(stamp, dict):
        return 0.0
    sec = float(stamp.get("sec", 0) or 0)
    nanosec = float(stamp.get("nanosec", 0) or 0)
    return sec + nanosec * 1e-9


def _latency_ms(entry: dict[str, object]) -> float | None:
    metadata = entry.get("metadata")
    if isinstance(metadata, dict) and metadata.get("latency_ms") is not None:
        return float(metadata["latency_ms"])
    return None


def load_action_log(path: Path) -> list[ReplayFrame]:
    """Parse a vla_zoo ``vla_actions.jsonl`` file into replay frames."""

    frames: list[ReplayFrame] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        entry: dict[str, object] = json.loads(stripped)
        raw_data = entry.get("data")
        raw_names = entry.get("names")
        data = raw_data if isinstance(raw_data, list) else []
        names = raw_names if isinstance(raw_names, list) else []
        frames.append(
            ReplayFrame(
                index=index,
                stamp_sec=_stamp_seconds(entry),
                model_name=str(entry.get("model_name", "")),
                action_space=str(entry.get("action_space", "custom")),
                action=tuple(float(value) for value in data),
                names=tuple(str(name) for name in names),
                latency_ms=_latency_ms(entry),
            )
        )
    return frames


def replay_action_rate_hz(frames: Sequence[ReplayFrame]) -> float | None:
    """Estimate action rate (Hz) from recorded frame timestamps."""

    stamps = [frame.stamp_sec for frame in frames if frame.stamp_sec > 0]
    if len(stamps) < 2:
        return None
    span = max(stamps) - min(stamps)
    if span <= 0:
        return None
    return (len(stamps) - 1) / span


def frames_to_records(
    frames: Sequence[ReplayFrame],
    *,
    task_id: str = "",
    source: str = REPLAY_SOURCE,
) -> list[EpisodeRecord]:
    """Map replay frames to schema episode records (success is always ``None``)."""

    return [
        EpisodeRecord(
            model=frame.model_name,
            source=source,
            index=frame.index,
            task_id=task_id,
            success=None,  # a recorded action stream is not a task-success claim
            latency_ms=frame.latency_ms,
            num_actions=len(frame.action),
            note=None,
        )
        for frame in frames
    ]
