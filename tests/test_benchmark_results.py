from __future__ import annotations

from pathlib import Path

import pytest

from vla_zoo.benchmark.results import (
    RESULT_SCHEMA_VERSION,
    EpisodeRecord,
    SchemaVersionError,
    format_benchmark_summary_markdown,
    read_episode_jsonl,
    summarize_records,
    write_episode_jsonl,
)
from vla_zoo.benchmark.runner import run_smoke_episode_records
from vla_zoo.core.registry import load_model


def _records() -> list[EpisodeRecord]:
    return [
        EpisodeRecord(
            model="dummy",
            source="smoke-benchmark",
            index=i,
            task_id=str(i),
            success=i % 2 == 0,
            latency_ms=float(i + 1),
            num_actions=1,
        )
        for i in range(4)
    ]


def test_episode_record_jsonl_roundtrip(tmp_path: Path) -> None:
    records = _records()
    path = write_episode_jsonl(tmp_path / "results.jsonl", records)
    restored = read_episode_jsonl(path)

    assert restored == records
    assert all(r.schema_version == RESULT_SCHEMA_VERSION for r in restored)


def test_read_episode_jsonl_rejects_bad_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"schema_version": "other/v9", "model": "x", "source": "s", '
                    '"index": 0, "task_id": "", "success": null, "latency_ms": null, '
                    '"num_actions": 0, "error": null, "note": null}\n', encoding="utf-8")

    with pytest.raises(SchemaVersionError):
        read_episode_jsonl(path)


def test_summarize_records_computes_latency_percentiles() -> None:
    summary = summarize_records(_records(), action_rate_hz=10.0)

    # latencies are [1.0, 2.0, 3.0, 4.0]
    assert summary.sample_count == 4
    assert summary.success_rate == 0.5
    assert summary.latency_ms_p50 == 2.0  # index int(0.5 * 3) == 1
    assert summary.latency_ms_p95 == 3.0  # index int(0.95 * 3) == 2
    assert summary.latency_ms_mean == 2.5
    assert summary.action_rate_hz == 10.0
    assert summary.exception_count == 0


def test_summarize_records_handles_no_success_claim() -> None:
    records = [
        EpisodeRecord(
            model="dummy",
            source="ros2-action-replay",
            index=i,
            task_id="",
            success=None,
            latency_ms=1.0,
            num_actions=7,
        )
        for i in range(3)
    ]
    summary = summarize_records(records)

    assert summary.success_rate is None  # honest: no task-success claim
    assert summary.sample_count == 3


def test_summary_markdown_is_runtime_centric() -> None:
    markdown = format_benchmark_summary_markdown(summarize_records(_records(), action_rate_hz=5.0))

    assert "Action rate" in markdown
    assert "not model task-success quality" in markdown


def test_run_smoke_episode_records_emits_schema() -> None:
    model = load_model("dummy")
    records, action_rate_hz = run_smoke_episode_records(model, model_name="dummy", episodes=3)

    assert len(records) == 3
    assert all(r.source == "smoke-benchmark" for r in records)
    assert all(r.schema_version == RESULT_SCHEMA_VERSION for r in records)
    assert action_rate_hz is None or action_rate_hz > 0
