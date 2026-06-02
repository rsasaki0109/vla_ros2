from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vla_zoo.core.types import ActionSpec, VLAAction
from vla_zoo.runtime.diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    RuntimeDiagnostics,
    SchemaVersionError,
    format_diagnostics_markdown,
    read_diagnostics_jsonl,
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
