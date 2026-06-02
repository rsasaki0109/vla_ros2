"""Versioned JSONL benchmark result schema and latency/action-rate summaries.

This module defines the canonical, machine-readable result format that benchmark
runners and replay loaders emit. It is intentionally runtime-centric: a row records
what the runtime did (latency, action count, success flag when a task defines one),
not a claim about model quality. ``success`` may be ``None`` when the source cannot
assert task success (for example, replayed action logs).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

#: Schema identifier embedded in every record and summary. Bump the version when the
#: field set changes in a backward-incompatible way.
RESULT_SCHEMA_VERSION = "vla-zoo-benchmark/v1"


class SchemaVersionError(ValueError):
    """Raised when a JSONL record does not carry the expected schema version."""


@dataclass(frozen=True)
class EpisodeRecord:
    """One benchmark sample: a benchmark episode or a single replayed action frame."""

    model: str
    source: str
    index: int
    task_id: str
    success: bool | None
    latency_ms: float | None
    num_actions: int
    error: str | None = None
    note: str | None = None
    schema_version: str = RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "source": self.source,
            "index": self.index,
            "task_id": self.task_id,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "num_actions": self.num_actions,
            "error": self.error,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> EpisodeRecord:
        version = payload.get("schema_version")
        if version != RESULT_SCHEMA_VERSION:
            msg = f"Unsupported schema_version {version!r}; expected {RESULT_SCHEMA_VERSION!r}"
            raise SchemaVersionError(msg)
        success = payload.get("success")
        latency = payload.get("latency_ms")
        return cls(
            model=str(payload["model"]),
            source=str(payload["source"]),
            index=int(str(payload["index"])),
            task_id=str(payload.get("task_id", "")),
            success=None if success is None else bool(success),
            latency_ms=None if latency is None else float(str(latency)),
            num_actions=int(str(payload.get("num_actions", 0))),
            error=None if payload.get("error") is None else str(payload["error"]),
            note=None if payload.get("note") is None else str(payload["note"]),
            schema_version=str(version),
        )


def write_episode_jsonl(path: Path, records: Iterable[EpisodeRecord]) -> Path:
    """Write episode records as one JSON object per line and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def read_episode_jsonl(path: Path) -> list[EpisodeRecord]:
    """Read and validate episode records from a JSONL file."""

    records: list[EpisodeRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(EpisodeRecord.from_dict(json.loads(stripped)))
    return records


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate latency / action-rate report computed from episode records."""

    model: str
    source: str
    sample_count: int
    success_count: int
    success_rate: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    latency_ms_mean: float | None
    action_rate_hz: float | None
    exception_count: int
    note: str | None = None
    schema_version: str = RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "source": self.source,
            "sample_count": self.sample_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_p95": self.latency_ms_p95,
            "latency_ms_mean": self.latency_ms_mean,
            "action_rate_hz": self.action_rate_hz,
            "exception_count": self.exception_count,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BenchmarkSummary:
        version = payload.get("schema_version")
        if version != RESULT_SCHEMA_VERSION:
            msg = f"Unsupported schema_version {version!r}; expected {RESULT_SCHEMA_VERSION!r}"
            raise SchemaVersionError(msg)

        def _opt_float(key: str) -> float | None:
            value = payload.get(key)
            return None if value is None else float(str(value))

        success_rate = payload.get("success_rate")
        return cls(
            model=str(payload["model"]),
            source=str(payload["source"]),
            sample_count=int(str(payload.get("sample_count", 0))),
            success_count=int(str(payload.get("success_count", 0))),
            success_rate=None if success_rate is None else float(str(success_rate)),
            latency_ms_p50=_opt_float("latency_ms_p50"),
            latency_ms_p95=_opt_float("latency_ms_p95"),
            latency_ms_mean=_opt_float("latency_ms_mean"),
            action_rate_hz=_opt_float("action_rate_hz"),
            exception_count=int(str(payload.get("exception_count", 0))),
            note=None if payload.get("note") is None else str(payload["note"]),
            schema_version=str(version),
        )


def read_summary_json(path: Path) -> BenchmarkSummary:
    """Read and validate a benchmark summary JSON file."""

    return BenchmarkSummary.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = int(fraction * (len(ordered) - 1))
    return ordered[index]


def summarize_records(
    records: Sequence[EpisodeRecord],
    *,
    action_rate_hz: float | None = None,
    note: str | None = None,
) -> BenchmarkSummary:
    """Reduce episode records to a latency / action-rate summary.

    ``success_rate`` is ``None`` when no record carries a boolean success flag (for
    example, replayed action logs that make no task-success claim). ``action_rate_hz``
    is passed in by the caller, which knows the real wall-clock or recorded timing.
    """

    if not records:
        return BenchmarkSummary(
            model="",
            source="",
            sample_count=0,
            success_count=0,
            success_rate=None,
            latency_ms_p50=None,
            latency_ms_p95=None,
            latency_ms_mean=None,
            action_rate_hz=action_rate_hz,
            exception_count=0,
            note=note,
        )

    latencies = [r.latency_ms for r in records if r.latency_ms is not None]
    successes = [r.success for r in records if r.success is not None]
    success_count = sum(1 for value in successes if value)
    return BenchmarkSummary(
        model=records[0].model,
        source=records[0].source,
        sample_count=len(records),
        success_count=success_count,
        success_rate=(success_count / len(successes)) if successes else None,
        latency_ms_p50=_percentile(latencies, 0.5) if latencies else None,
        latency_ms_p95=_percentile(latencies, 0.95) if latencies else None,
        latency_ms_mean=mean(latencies) if latencies else None,
        action_rate_hz=action_rate_hz,
        exception_count=sum(1 for r in records if r.error is not None),
        note=note,
    )


def _fmt(value: float | None, *, suffix: str = "") -> str:
    return f"{value:.2f}{suffix}" if value is not None else "-"


def format_benchmark_summary_markdown(
    summary: BenchmarkSummary,
    *,
    title: str = "Benchmark Latency / Action-Rate Summary",
) -> str:
    """Render a latency / action-rate summary report as Markdown."""

    success = "-" if summary.success_rate is None else f"{summary.success_rate:.2%}"
    lines = [
        f"# {title}",
        "",
        f"- Schema: `{summary.schema_version}`",
        f"- Source: `{summary.source}`",
        f"- Model: `{summary.model}`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Samples | {summary.sample_count} |",
        f"| Success rate | {success} |",
        f"| Latency p50 | {_fmt(summary.latency_ms_p50, suffix=' ms')} |",
        f"| Latency p95 | {_fmt(summary.latency_ms_p95, suffix=' ms')} |",
        f"| Latency mean | {_fmt(summary.latency_ms_mean, suffix=' ms')} |",
        f"| Action rate | {_fmt(summary.action_rate_hz, suffix=' Hz')} |",
        f"| Exceptions | {summary.exception_count} |",
        "",
        (
            summary.note
            or "Runtime-centric summary. It measures latency and action throughput, "
            "not model task-success quality."
        ),
        "",
    ]
    return "\n".join(lines) + "\n"
