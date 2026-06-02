from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vla_zoo.core.types import ActionSpec, VLAAction
from vla_zoo.runtime.diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    RuntimeDiagnostics,
    SchemaVersionError,
    diagnostics_from_key_values,
    format_diagnostics_markdown,
    format_diagnostics_summary_markdown,
    read_diagnostics_jsonl,
    summarize_diagnostics,
    write_diagnostics_jsonl,
)
from vla_zoo.runtime.guard import ActionClipGuard, evaluate_watchdog


def _action(values: list[float]) -> VLAAction:
    spec = ActionSpec(
        action_space="eef_delta",
        shape=(3,),
        low=(-1.0, -1.0, -1.0),
        high=(1.0, 1.0, 1.0),
    )
    return VLAAction(data=np.asarray(values, dtype=np.float32), spec=spec)


def _record() -> RuntimeDiagnostics:
    guard = ActionClipGuard()
    guard.clip(_action([2.0, 0.0, 0.0]))  # clipped
    guard.clip(_action([0.0, 0.0, 0.0]))  # not clipped
    watchdog = evaluate_watchdog(image_age_sec=0.1, instruction_age_sec=0.2)
    return RuntimeDiagnostics.from_parts(
        model="dummy",
        status_text="ready",
        level="ok",
        clip_guard=guard,
        watchdog=watchdog,
        last_latency_ms=12.5,
        avg_latency_ms=11.0,
        action_rate_hz=5.0,
        dropped_frames=1,
        pending_inference=False,
    )


def test_from_parts_merges_clip_guard_and_watchdog() -> None:
    record = _record()

    assert record.total_actions == 2
    assert record.clipped_actions == 1
    assert record.action_clip_rate == 0.5
    assert record.watchdog_ok is True
    assert record.watchdog_reason is None
    assert record.image_age_sec == 0.1
    assert record.schema_version == DIAGNOSTICS_SCHEMA_VERSION


def test_from_parts_carries_watchdog_reason() -> None:
    watchdog = evaluate_watchdog(image_age_sec=None, instruction_age_sec=0.1)
    record = RuntimeDiagnostics.from_parts(
        model="dummy",
        status_text="waiting for image",
        level="warn",
        clip_guard=ActionClipGuard(),
        watchdog=watchdog,
        last_latency_ms=None,
        avg_latency_ms=None,
    )

    assert record.watchdog_ok is False
    assert record.watchdog_reason == "waiting for image"
    assert record.level == "warn"


def test_round_trip_dict() -> None:
    record = _record()

    restored = RuntimeDiagnostics.from_dict(record.to_dict())

    assert restored == record


def test_from_dict_rejects_unknown_schema() -> None:
    payload = _record().to_dict()
    payload["schema_version"] = "vla-zoo-diagnostics/v999"

    with pytest.raises(SchemaVersionError):
        RuntimeDiagnostics.from_dict(payload)


def test_jsonl_round_trip(tmp_path: Path) -> None:
    records = [_record(), _record()]
    path = write_diagnostics_jsonl(tmp_path / "diag.jsonl", records)

    restored = read_diagnostics_jsonl(path)

    assert restored == records


def test_to_key_values_blanks_missing_latency() -> None:
    record = RuntimeDiagnostics.from_parts(
        model="dummy",
        status_text="waiting for instruction",
        level="warn",
        clip_guard=ActionClipGuard(),
        watchdog=evaluate_watchdog(image_age_sec=0.1, instruction_age_sec=0.2),
        last_latency_ms=None,
        avg_latency_ms=None,
    )

    pairs = dict(record.to_key_values())

    assert pairs["last_latency_ms"] == ""
    assert pairs["schema_version"] == DIAGNOSTICS_SCHEMA_VERSION
    assert pairs["action_clip_rate"] == "0.0000"


def test_format_markdown_contains_metrics() -> None:
    text = format_diagnostics_markdown(_record())

    assert "Runtime Diagnostics Snapshot" in text
    assert DIAGNOSTICS_SCHEMA_VERSION in text
    assert "Action clip rate" in text
    assert "not model task-success quality" in text


def test_diagnostics_from_key_values_round_trips() -> None:
    record = _record()

    restored = diagnostics_from_key_values(record.to_key_values())

    # KeyValue pairs are strings rounded to a few decimals, so rates round-trip approximately.
    assert restored.model == record.model
    assert restored.status_text == record.status_text
    assert restored.total_actions == record.total_actions
    assert restored.clipped_actions == record.clipped_actions
    assert restored.watchdog_ok == record.watchdog_ok
    assert restored.last_latency_ms == pytest.approx(record.last_latency_ms, abs=1e-3)
    assert restored.action_clip_rate == pytest.approx(record.action_clip_rate, abs=1e-4)
    assert restored.element_clip_rate == pytest.approx(record.element_clip_rate, abs=1e-4)


def test_diagnostics_from_key_values_parses_string_bools_and_blanks() -> None:
    record = RuntimeDiagnostics.from_parts(
        model="smolvla",
        status_text="waiting for instruction",
        level="warn",
        clip_guard=ActionClipGuard(),
        watchdog=evaluate_watchdog(image_age_sec=0.1, instruction_age_sec=0.2),
        last_latency_ms=None,
        avg_latency_ms=None,
        pending_inference=False,
    )

    restored = diagnostics_from_key_values(record.to_key_values())

    # bool("False") is True, so naive from_dict would corrupt these — the inverse must not.
    assert restored.pending_inference is False
    assert restored.watchdog_ok is True
    assert restored.last_latency_ms is None
    assert restored == record


def _seq_record(*, level: str, latency: float | None, dropped: int) -> RuntimeDiagnostics:
    return RuntimeDiagnostics.from_parts(
        model="smolvla",
        status_text="ok" if level == "ok" else "waiting for instruction",
        level=level,
        clip_guard=ActionClipGuard(),
        watchdog=evaluate_watchdog(image_age_sec=0.1, instruction_age_sec=0.1),
        last_latency_ms=latency,
        avg_latency_ms=latency,
        dropped_frames=dropped,
    )


def test_summarize_diagnostics_reduces_latency_and_worst_level() -> None:
    records = [
        _seq_record(level="warn", latency=None, dropped=0),
        _seq_record(level="ok", latency=100.0, dropped=5),
        _seq_record(level="ok", latency=300.0, dropped=12),
        _seq_record(level="ok", latency=200.0, dropped=12),
    ]

    summary = summarize_diagnostics(records)

    assert summary.record_count == 4
    assert summary.latency_ms_min == 100.0
    assert summary.latency_ms_max == 300.0
    assert summary.latency_ms_p50 == 200.0
    assert summary.max_dropped_frames == 12
    # the transient warn must surface even though the final record is ok
    assert summary.worst_level == "warn"
    assert summary.worst_index == 0


def test_summarize_diagnostics_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarize_diagnostics([])


def test_format_summary_markdown_is_runtime_centric() -> None:
    summary = summarize_diagnostics([_seq_record(level="ok", latency=50.0, dropped=1)])

    text = format_diagnostics_summary_markdown(summary)

    assert "Runtime Diagnostics Summary" in text
    assert DIAGNOSTICS_SCHEMA_VERSION in text
    assert "not model task-success quality" in text
