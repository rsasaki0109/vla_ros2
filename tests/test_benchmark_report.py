from __future__ import annotations

import json
from pathlib import Path

import pytest

from vla_zoo.benchmark.report import (
    format_benchmark_report_html,
    format_benchmark_report_markdown,
)
from vla_zoo.benchmark.results import (
    RESULT_SCHEMA_VERSION,
    BenchmarkSummary,
    SchemaVersionError,
    read_summary_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SUMMARY = REPO_ROOT / "docs" / "assets" / "sample_benchmark" / "ros2_replay_summary.json"


def _summary(model: str, *, success_rate: float | None) -> BenchmarkSummary:
    return BenchmarkSummary(
        model=model,
        source="smoke-benchmark",
        sample_count=5,
        success_count=5 if success_rate else 0,
        success_rate=success_rate,
        latency_ms_p50=1.0,
        latency_ms_p95=2.0,
        latency_ms_mean=1.5,
        action_rate_hz=10.0,
        exception_count=0,
    )


def test_summary_json_roundtrip(tmp_path: Path) -> None:
    summary = _summary("dummy", success_rate=1.0)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary.to_dict()), encoding="utf-8")

    assert read_summary_json(path) == summary


def test_read_summary_json_rejects_bad_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": "other/v2", "model": "x", "source": "s"}),
                    encoding="utf-8")

    with pytest.raises(SchemaVersionError):
        read_summary_json(path)


def test_markdown_report_lists_each_summary() -> None:
    markdown = format_benchmark_report_markdown(
        [_summary("dummy", success_rate=1.0), _summary("scripted", success_rate=None)],
        title="Compare",
    )

    assert "# Compare" in markdown
    assert "dummy" in markdown
    assert "scripted" in markdown
    assert "not robot task-success quality" in markdown


def test_html_report_is_standalone_and_escapes() -> None:
    html = format_benchmark_report_html([_summary("dummy", success_rate=None)], title="Compare")

    assert html.startswith("<!doctype html>")
    assert RESULT_SCHEMA_VERSION in html
    assert "ros2-action-replay" not in html  # only smoke summary rendered
    assert "dummy" in html


def test_sample_benchmark_report_artifacts_exist() -> None:
    assert SAMPLE_SUMMARY.is_file()
    assert (REPO_ROOT / "docs" / "assets" / "sample_benchmark" / "benchmark_report.html").is_file()
    # the checked-in sample summary must be readable under the current schema
    read_summary_json(SAMPLE_SUMMARY)
