"""Versioned, pure runtime diagnostics record: clip-rate + watchdog + latency in one schema.

The runtime guards (``ActionClipGuard`` clip-rate counters and the ``WatchdogStatus``
staleness flag) and latency were previously reported ad hoc by the ROS2 node. This module
merges them into one canonical, machine-readable record with a JSONL/Markdown surface,
mirroring ``benchmark/results.py``. The ROS2 node's ``/diagnostics`` payload is built from
this record so the published key/values, the JSONL log, and the Markdown report all share
one definition.

It is runtime-centric: a record describes what the runtime did (latency, clip rate,
staleness) — never a claim about model task-success quality. The record carries no side
effects and no heavy dependencies, so it is unit-testable directly.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from vla_zoo.runtime.guard import ActionClipGuard, WatchdogStatus

#: Schema identifier embedded in every diagnostics record. Bump the version when the field
#: set changes in a backward-incompatible way.
DIAGNOSTICS_SCHEMA_VERSION = "vla-zoo-diagnostics/v1"

#: Diagnostic severity levels, matching the ROS2 ``diagnostic_msgs`` ordering.
DIAGNOSTIC_LEVELS = ("ok", "warn", "error")


class SchemaVersionError(ValueError):
    """Raised when a diagnostics record does not carry the expected schema version."""


@dataclass(frozen=True)
class RuntimeDiagnostics:
    """One runtime diagnostics snapshot: clip-rate, staleness watchdog, and latency.

    This is the single source of truth for the ROS2 node's ``/diagnostics`` payload, a
    JSONL diagnostics log, and a Markdown report. ``level`` is one of ``DIAGNOSTIC_LEVELS``.
    """

    model: str
    status_text: str
    level: str
    last_latency_ms: float | None
    avg_latency_ms: float | None
    action_rate_hz: float | None
    dropped_frames: int
    pending_inference: bool
    total_actions: int
    clipped_actions: int
    clipped_elements: int
    total_elements: int
    action_clip_rate: float
    element_clip_rate: float
    watchdog_ok: bool
    watchdog_reason: str | None
    image_age_sec: float | None
    instruction_age_sec: float | None
    note: str | None = None
    schema_version: str = DIAGNOSTICS_SCHEMA_VERSION

    @classmethod
    def from_parts(
        cls,
        *,
        model: str,
        status_text: str,
        level: str,
        clip_guard: ActionClipGuard,
        watchdog: WatchdogStatus,
        last_latency_ms: float | None,
        avg_latency_ms: float | None,
        action_rate_hz: float | None = None,
        dropped_frames: int = 0,
        pending_inference: bool = False,
        note: str | None = None,
    ) -> RuntimeDiagnostics:
        """Build a record by merging the clip guard counters and the watchdog status."""

        return cls(
            model=model,
            status_text=status_text,
            level=level,
            last_latency_ms=last_latency_ms,
            avg_latency_ms=avg_latency_ms,
            action_rate_hz=action_rate_hz,
            dropped_frames=dropped_frames,
            pending_inference=pending_inference,
            total_actions=clip_guard.total_actions,
            clipped_actions=clip_guard.clipped_actions,
            clipped_elements=clip_guard.clipped_elements,
            total_elements=clip_guard.total_elements,
            action_clip_rate=clip_guard.action_clip_rate,
            element_clip_rate=clip_guard.element_clip_rate,
            watchdog_ok=watchdog.ok,
            watchdog_reason=watchdog.reason,
            image_age_sec=watchdog.image_age_sec,
            instruction_age_sec=watchdog.instruction_age_sec,
            note=note,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "status_text": self.status_text,
            "level": self.level,
            "last_latency_ms": self.last_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "action_rate_hz": self.action_rate_hz,
            "dropped_frames": self.dropped_frames,
            "pending_inference": self.pending_inference,
            "total_actions": self.total_actions,
            "clipped_actions": self.clipped_actions,
            "clipped_elements": self.clipped_elements,
            "total_elements": self.total_elements,
            "action_clip_rate": self.action_clip_rate,
            "element_clip_rate": self.element_clip_rate,
            "watchdog_ok": self.watchdog_ok,
            "watchdog_reason": self.watchdog_reason,
            "image_age_sec": self.image_age_sec,
            "instruction_age_sec": self.instruction_age_sec,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> RuntimeDiagnostics:
        version = payload.get("schema_version")
        if version != DIAGNOSTICS_SCHEMA_VERSION:
            msg = (
                f"Unsupported schema_version {version!r}; "
                f"expected {DIAGNOSTICS_SCHEMA_VERSION!r}"
            )
            raise SchemaVersionError(msg)

        def _opt_float(key: str) -> float | None:
            value = payload.get(key)
            return None if value is None else float(str(value))

        def _opt_str(key: str) -> str | None:
            value = payload.get(key)
            return None if value is None else str(value)

        return cls(
            model=str(payload["model"]),
            status_text=str(payload.get("status_text", "")),
            level=str(payload.get("level", "ok")),
            last_latency_ms=_opt_float("last_latency_ms"),
            avg_latency_ms=_opt_float("avg_latency_ms"),
            action_rate_hz=_opt_float("action_rate_hz"),
            dropped_frames=int(str(payload.get("dropped_frames", 0))),
            pending_inference=bool(payload.get("pending_inference", False)),
            total_actions=int(str(payload.get("total_actions", 0))),
            clipped_actions=int(str(payload.get("clipped_actions", 0))),
            clipped_elements=int(str(payload.get("clipped_elements", 0))),
            total_elements=int(str(payload.get("total_elements", 0))),
            action_clip_rate=float(str(payload.get("action_clip_rate", 0.0))),
            element_clip_rate=float(str(payload.get("element_clip_rate", 0.0))),
            watchdog_ok=bool(payload.get("watchdog_ok", True)),
            watchdog_reason=_opt_str("watchdog_reason"),
            image_age_sec=_opt_float("image_age_sec"),
            instruction_age_sec=_opt_float("instruction_age_sec"),
            note=_opt_str("note"),
            schema_version=str(version),
        )

    def to_key_values(self) -> list[tuple[str, str]]:
        """Flatten to ROS2 ``diagnostic_msgs/KeyValue`` string pairs (key, value)."""

        def _num(value: float | None, *, fmt: str) -> str:
            return "" if value is None else format(value, fmt)

        return [
            ("schema_version", self.schema_version),
            ("model", self.model),
            ("status_text", self.status_text),
            ("level", self.level),
            ("last_latency_ms", _num(self.last_latency_ms, fmt=".3f")),
            ("avg_latency_ms", _num(self.avg_latency_ms, fmt=".3f")),
            ("action_rate_hz", _num(self.action_rate_hz, fmt=".3f")),
            ("dropped_frames", str(self.dropped_frames)),
            ("pending_inference", str(self.pending_inference)),
            ("total_actions", str(self.total_actions)),
            ("clipped_actions", str(self.clipped_actions)),
            ("clipped_elements", str(self.clipped_elements)),
            ("total_elements", str(self.total_elements)),
            ("action_clip_rate", f"{self.action_clip_rate:.4f}"),
            ("element_clip_rate", f"{self.element_clip_rate:.4f}"),
            ("watchdog_ok", str(self.watchdog_ok)),
            ("watchdog_reason", self.watchdog_reason or ""),
            ("image_age_sec", _num(self.image_age_sec, fmt=".3f")),
            ("instruction_age_sec", _num(self.instruction_age_sec, fmt=".3f")),
        ]


def write_diagnostics_jsonl(path: Path, records: Iterable[RuntimeDiagnostics]) -> Path:
    """Write diagnostics records as one JSON object per line and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def read_diagnostics_jsonl(path: Path) -> list[RuntimeDiagnostics]:
    """Read and validate diagnostics records from a JSONL file."""

    records: list[RuntimeDiagnostics] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(RuntimeDiagnostics.from_dict(json.loads(stripped)))
    return records


def diagnostics_from_key_values(
    pairs: Mapping[str, str] | Iterable[tuple[str, str]],
) -> RuntimeDiagnostics:
    """Rebuild a record from flattened ``(key, value)`` string pairs.

    This is the inverse of :meth:`RuntimeDiagnostics.to_key_values`. Use it to recover the
    native record from a recorded ROS2 ``diagnostic_msgs/KeyValue`` payload, where every
    value is a string and an empty string stands for a missing optional number. ``from_dict``
    cannot be used for that source because Python's ``bool("False")`` is ``True``.
    """

    data = dict(pairs)

    def _opt_float(key: str) -> float | None:
        value = data.get(key, "")
        return float(value) if value not in ("", None) else None

    def _bool(key: str) -> bool:
        return str(data.get(key, "")).strip().lower() == "true"

    def _opt_str(key: str) -> str | None:
        value = data.get(key)
        return value if value else None

    return RuntimeDiagnostics(
        model=str(data.get("model", "")),
        status_text=str(data.get("status_text", "")),
        level=str(data.get("level", "ok")),
        last_latency_ms=_opt_float("last_latency_ms"),
        avg_latency_ms=_opt_float("avg_latency_ms"),
        action_rate_hz=_opt_float("action_rate_hz"),
        dropped_frames=int(data.get("dropped_frames", "0") or "0"),
        pending_inference=_bool("pending_inference"),
        total_actions=int(data.get("total_actions", "0") or "0"),
        clipped_actions=int(data.get("clipped_actions", "0") or "0"),
        clipped_elements=int(data.get("clipped_elements", "0") or "0"),
        total_elements=int(data.get("total_elements", "0") or "0"),
        action_clip_rate=float(data.get("action_clip_rate", "0.0") or "0.0"),
        element_clip_rate=float(data.get("element_clip_rate", "0.0") or "0.0"),
        watchdog_ok=_bool("watchdog_ok"),
        watchdog_reason=_opt_str("watchdog_reason"),
        image_age_sec=_opt_float("image_age_sec"),
        instruction_age_sec=_opt_float("instruction_age_sec"),
        note=_opt_str("note"),
        schema_version=str(data.get("schema_version", DIAGNOSTICS_SCHEMA_VERSION)),
    )


def _fmt(value: float | None, *, suffix: str = "") -> str:
    return f"{value:.2f}{suffix}" if value is not None else "-"


def format_diagnostics_markdown(
    record: RuntimeDiagnostics,
    *,
    title: str = "Runtime Diagnostics Snapshot",
) -> str:
    """Render a diagnostics record as Markdown."""

    watchdog = "ok" if record.watchdog_ok else (record.watchdog_reason or "not ok")
    lines = [
        f"# {title}",
        "",
        f"- Schema: `{record.schema_version}`",
        f"- Model: `{record.model}`",
        f"- Level: `{record.level}`",
        f"- Status: {record.status_text}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Last latency | {_fmt(record.last_latency_ms, suffix=' ms')} |",
        f"| Avg latency | {_fmt(record.avg_latency_ms, suffix=' ms')} |",
        f"| Action rate | {_fmt(record.action_rate_hz, suffix=' Hz')} |",
        f"| Dropped frames | {record.dropped_frames} |",
        f"| Pending inference | {record.pending_inference} |",
        f"| Actions seen | {record.total_actions} |",
        f"| Clipped actions | {record.clipped_actions} |",
        f"| Action clip rate | {record.action_clip_rate:.2%} |",
        f"| Element clip rate | {record.element_clip_rate:.2%} |",
        f"| Watchdog | {watchdog} |",
        f"| Image age | {_fmt(record.image_age_sec, suffix=' s')} |",
        f"| Instruction age | {_fmt(record.instruction_age_sec, suffix=' s')} |",
        "",
        (
            record.note
            or "Runtime-centric diagnostics. It reports latency, clip rate, and input "
            "staleness, not model task-success quality."
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class DiagnosticsSummary:
    """A time-series reduction of a full diagnostics log into one runtime view."""

    record_count: int
    model: str
    latency_ms_min: float | None
    latency_ms_p50: float | None
    latency_ms_max: float | None
    max_dropped_frames: int
    final_total_actions: int
    peak_action_clip_rate: float
    peak_element_clip_rate: float
    worst_level: str
    worst_status_text: str
    worst_index: int
    schema_version: str = DIAGNOSTICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "record_count": self.record_count,
            "model": self.model,
            "latency_ms_min": self.latency_ms_min,
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_max": self.latency_ms_max,
            "max_dropped_frames": self.max_dropped_frames,
            "final_total_actions": self.final_total_actions,
            "peak_action_clip_rate": self.peak_action_clip_rate,
            "peak_element_clip_rate": self.peak_element_clip_rate,
            "worst_level": self.worst_level,
            "worst_status_text": self.worst_status_text,
            "worst_index": self.worst_index,
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = int(fraction * (len(ordered) - 1))
    return ordered[index]


def _level_rank(level: str) -> int:
    try:
        return DIAGNOSTIC_LEVELS.index(level)
    except ValueError:
        return 0


def summarize_diagnostics(records: Sequence[RuntimeDiagnostics]) -> DiagnosticsSummary:
    """Reduce a diagnostics log to latency spread, clip-rate peaks, and the worst record.

    Latency is aggregated over each record's ``last_latency_ms`` (``None`` values skipped).
    The worst record is the highest ``level`` (ties resolved to the latest occurrence), so a
    transient ``warn``/``error`` is never hidden by a final ``ok`` snapshot. It stays a
    runtime-path view: latency, drops, clip rate, staleness — never task-success quality.
    """

    if not records:
        msg = "cannot summarize an empty diagnostics log"
        raise ValueError(msg)

    latencies = [r.last_latency_ms for r in records if r.last_latency_ms is not None]
    worst_index = max(
        range(len(records)),
        key=lambda i: (_level_rank(records[i].level), i),
    )
    worst = records[worst_index]

    return DiagnosticsSummary(
        record_count=len(records),
        model=records[-1].model,
        latency_ms_min=min(latencies) if latencies else None,
        latency_ms_p50=_percentile(latencies, 0.5) if latencies else None,
        latency_ms_max=max(latencies) if latencies else None,
        max_dropped_frames=max(r.dropped_frames for r in records),
        final_total_actions=records[-1].total_actions,
        peak_action_clip_rate=max(r.action_clip_rate for r in records),
        peak_element_clip_rate=max(r.element_clip_rate for r in records),
        worst_level=worst.level,
        worst_status_text=worst.status_text,
        worst_index=worst_index,
    )


def format_diagnostics_summary_markdown(
    summary: DiagnosticsSummary,
    *,
    title: str = "Runtime Diagnostics Summary",
) -> str:
    """Render a time-series diagnostics summary as Markdown."""

    lines = [
        f"# {title}",
        "",
        f"- Schema: `{summary.schema_version}`",
        f"- Model: `{summary.model}`",
        f"- Records: {summary.record_count}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Latency min / p50 / max | {_fmt(summary.latency_ms_min)} / "
        f"{_fmt(summary.latency_ms_p50)} / {_fmt(summary.latency_ms_max, suffix=' ms')} |",
        f"| Max dropped frames | {summary.max_dropped_frames} |",
        f"| Final actions seen | {summary.final_total_actions} |",
        f"| Peak action clip rate | {summary.peak_action_clip_rate:.2%} |",
        f"| Peak element clip rate | {summary.peak_element_clip_rate:.2%} |",
        f"| Worst level | `{summary.worst_level}` (record #{summary.worst_index}) |",
        f"| Worst status | {summary.worst_status_text} |",
        "",
        (
            "Runtime-centric summary over the full log. It reports latency spread, drop and "
            "clip-rate peaks, and the worst-severity record, not model task-success quality."
        ),
        "",
    ]
    return "\n".join(lines) + "\n"
