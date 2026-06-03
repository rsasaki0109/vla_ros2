"""Record a VLA adapter's action stream on real PyBullet-rendered scene frames.

This is a *runtime-path* probe, not a benchmark and **not** a task-success or
policy-quality claim. It drives an adapter through the existing PyBullet pick-and-place
rollout and records, for every fresh adapter query, the full action vector the adapter
produced from a genuinely rendered camera frame — exercising the real image
preprocessing / encode path that the synthetic-random-frame probes skip.

The recorded log is written in vla_zoo's canonical ``vla_actions.jsonl`` shape, so it can
be replayed through :func:`vla_zoo.benchmark.replay.load_action_log` and summarized by
``bench-replay`` (which always records ``success=None``). The only thing this upgrades is
the *input*: from synthetic noise to a real scene render. The action semantics, and
whether they would actually accomplish the task, remain unverified.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vla_zoo.core.types import VLAAction, VLAActionChunk

if TYPE_CHECKING:
    from vla_zoo.demo.pybullet import AdapterPredictionEvent, PyBulletDemoConfig

#: Stable identifier for the probe input modality, recorded in every artifact.
PROBE_IMAGE_SOURCE = "pybullet-render"

#: The one honesty caveat reused across the summary, JSON, and Markdown.
PROBE_NOTE = (
    "Runtime evidence: a real adapter driven on real PyBullet-rendered scene frames, "
    "recording latency and action magnitude. It exercises the real image preprocessing "
    "path that synthetic-frame probes skip. It is NOT a task-success or policy-quality "
    "claim (policy_quality=not_verified): the action stream is not evidence that the "
    "robot task was completed."
)


def _resolved_action(prediction: VLAAction | VLAActionChunk) -> VLAAction:
    """The single executed action: the first action of a chunk, else the action itself."""

    return prediction.actions[0] if isinstance(prediction, VLAActionChunk) else prediction


def _stamp_from_seconds(seconds: float) -> dict[str, int]:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return {"sec": sec, "nanosec": max(0, nanosec)}


def build_probe_record(event: AdapterPredictionEvent) -> dict[str, Any]:
    """Turn one fresh adapter prediction into a ``vla_actions.jsonl``-compatible record.

    The shape matches what :func:`vla_zoo.benchmark.replay.load_action_log` reads (``data``,
    ``names``, ``model_name``, ``action_space``, ``header.stamp``, ``metadata.latency_ms``),
    with extra ``metadata`` keys documenting that the input was a real render.
    """

    action = _resolved_action(event.prediction)
    flat = action.to_numpy().reshape(-1)
    values = [float(value) for value in flat.tolist()]
    abs_values = np.abs(flat)
    names = list(action.spec.names) if action.spec.names else []
    chunk_size = (
        len(event.prediction.actions) if isinstance(event.prediction, VLAActionChunk) else 1
    )
    return {
        "header": {"stamp": _stamp_from_seconds(event.sim_time), "frame_id": "world"},
        "model_name": event.model_name,
        "action_space": action.spec.action_space,
        "data": values,
        "names": names,
        "confidence": action.confidence,
        "metadata": {
            "latency_ms": event.latency_ms,
            "phase": event.phase,
            "frame_index": event.frame_index,
            "query_index": event.query_index,
            "runtime": event.runtime,
            "instruction": event.instruction,
            "task_id": event.task_id,
            "image_source": PROBE_IMAGE_SOURCE,
            "camera_keys": list(event.camera_keys),
            "chunk_size": chunk_size,
            "abs_action_mean": float(abs_values.mean()) if abs_values.size else 0.0,
            "abs_action_max": float(abs_values.max()) if abs_values.size else 0.0,
            "real_scene": True,
            "policy_quality": "not_verified",
        },
    }


@dataclass(frozen=True)
class ActionProbeSummary:
    """Aggregate runtime-evidence statistics over a real-scene action probe."""

    model: str
    runtime: str
    instruction: str
    task_id: str
    image_source: str
    record_count: int
    action_dim: int
    latency_ms_min: float | None
    latency_ms_p50: float | None
    latency_ms_max: float | None
    latency_ms_mean: float | None
    abs_action_mean: float | None
    abs_action_max: float | None
    sample_action: tuple[float, ...]
    policy_quality: str = "not_verified"
    note: str = PROBE_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "runtime": self.runtime,
            "instruction": self.instruction,
            "task_id": self.task_id,
            "image_source": self.image_source,
            "record_count": self.record_count,
            "action_dim": self.action_dim,
            "latency_ms_min": self.latency_ms_min,
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_max": self.latency_ms_max,
            "latency_ms_mean": self.latency_ms_mean,
            "abs_action_mean": self.abs_action_mean,
            "abs_action_max": self.abs_action_max,
            "sample_action": list(self.sample_action),
            "policy_quality": self.policy_quality,
            "note": self.note,
        }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_probe_records(
    records: list[dict[str, Any]],
    *,
    model: str,
    runtime: str,
    instruction: str,
    task_id: str,
) -> ActionProbeSummary:
    """Reduce probe records into an :class:`ActionProbeSummary` (raises on an empty log)."""

    if not records:
        msg = "cannot summarize an empty action probe (no adapter queries recorded)"
        raise ValueError(msg)

    latencies = [
        float(record["metadata"]["latency_ms"])
        for record in records
        if record.get("metadata", {}).get("latency_ms") is not None
    ]
    abs_means = [
        float(record["metadata"]["abs_action_mean"])
        for record in records
        if record.get("metadata", {}).get("abs_action_mean") is not None
    ]
    abs_maxes = [
        float(record["metadata"]["abs_action_max"])
        for record in records
        if record.get("metadata", {}).get("abs_action_max") is not None
    ]
    last_data = [float(value) for value in records[-1].get("data", [])]
    action_dim = max((len(record.get("data", [])) for record in records), default=0)

    return ActionProbeSummary(
        model=model,
        runtime=runtime,
        instruction=instruction,
        task_id=task_id,
        image_source=PROBE_IMAGE_SOURCE,
        record_count=len(records),
        action_dim=action_dim,
        latency_ms_min=min(latencies) if latencies else None,
        latency_ms_p50=_percentile(latencies, 0.5) if latencies else None,
        latency_ms_max=max(latencies) if latencies else None,
        latency_ms_mean=statistics.fmean(latencies) if latencies else None,
        abs_action_mean=statistics.fmean(abs_means) if abs_means else None,
        abs_action_max=max(abs_maxes) if abs_maxes else None,
        sample_action=tuple(round(value, 6) for value in last_data),
    )


def write_action_probe_log(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write probe records as a ``vla_actions.jsonl`` action log (one JSON object per line)."""

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, separators=(",", ":")) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_action_probe_summary_markdown(
    summary: ActionProbeSummary,
    *,
    title: str = "Real-Scene Action Probe (Runtime Path)",
) -> str:
    sample = ", ".join(f"{value:.4f}" for value in summary.sample_action) or "n/a"
    lines = [
        f"# {title}",
        "",
        f"- Model: `{summary.model}` (runtime `{summary.runtime}`)",
        f"- Task: `{summary.task_id}` — \"{summary.instruction}\"",
        f"- Input: **{summary.image_source}** (real rendered scene frames, not synthetic noise)",
        f"- Adapter queries recorded: {summary.record_count}",
        f"- Action dim: {summary.action_dim}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Latency min | {_format_ms(summary.latency_ms_min)} |",
        f"| Latency p50 | {_format_ms(summary.latency_ms_p50)} |",
        f"| Latency max | {_format_ms(summary.latency_ms_max)} |",
        f"| Latency mean | {_format_ms(summary.latency_ms_mean)} |",
        f"| Mean abs action | {_format_float(summary.abs_action_mean)} |",
        f"| Max abs action | {_format_float(summary.abs_action_max)} |",
        f"| Sample action | {sample} |",
        f"| Policy quality | `{summary.policy_quality}` |",
        "",
        f"> {summary.note}",
        "",
    ]
    return "\n".join(lines)


def record_pybullet_action_probe(
    config: PyBulletDemoConfig,
) -> tuple[ActionProbeSummary, list[dict[str, Any]]]:
    """Run the PyBullet rollout and record the adapter's full action stream.

    Returns the runtime-evidence summary and the per-query ``vla_actions.jsonl`` records.
    Drives the real adapter on real rendered frames; makes no task-success claim.
    """

    from vla_zoo.demo.pybullet import run_simulation

    records: list[dict[str, Any]] = []

    def _sink(event: AdapterPredictionEvent) -> None:
        records.append(build_probe_record(event))

    run_simulation(config, prediction_sink=_sink)
    summary = summarize_probe_records(
        records,
        model=config.model_name,
        runtime=config.runtime,
        instruction=config.instruction,
        task_id=config.task_id,
    )
    return summary, records
