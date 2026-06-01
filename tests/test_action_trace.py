from __future__ import annotations

import json
from pathlib import Path

from vla_zoo.runtime.action_trace import (
    analyze_action_trace,
    format_action_analysis_markdown,
    format_action_trace_html,
    load_action_trace_events,
    summarize_action_trace,
)


def test_load_action_trace_events(tmp_path: Path) -> None:
    log = tmp_path / "actions.jsonl"
    log.write_text(
        json.dumps(
            {
                "header": {"stamp": {"sec": 10, "nanosec": 0}, "frame_id": "base_link"},
                "model_name": "dummy",
                "adapter_name": "DummyAdapter",
                "action_space": "eef_delta",
                "data": [0.0, 0.1, -0.2],
                "names": ["x", "y", "z"],
                "confidence": 1.0,
                "dt": 0.2,
                "chunk_index": 0,
                "metadata": {"task_id": "trace_test"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "header": {"stamp": {"sec": 11, "nanosec": 500_000_000}, "frame_id": "base_link"},
                "model_name": "dummy",
                "adapter_name": "DummyAdapter",
                "action_space": "eef_delta",
                "data": [0.2, 0.0, 0.0],
                "names": ["x", "y", "z"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    events = load_action_trace_events(log)
    summary = summarize_action_trace(events)

    assert len(events) == 2
    assert events[0].relative_sec == 0.0
    assert events[1].relative_sec == 1.5
    assert summary.action_count == 2
    assert summary.model_names == ["dummy"]
    assert summary.action_spaces == ["eef_delta"]
    assert summary.max_magnitude == 0.30000000000000004


def test_analyze_action_trace_flags_repeated_zero_actions(tmp_path: Path) -> None:
    log = tmp_path / "actions.jsonl"
    rows = []
    for index in range(4):
        rows.append(
            {
                "header": {"stamp": {"sec": 10 + index, "nanosec": 0}},
                "model_name": "dummy",
                "adapter_name": "DummyAdapter",
                "action_space": "eef_delta",
                "data": [0.0, 0.0],
                "names": ["x", "y"],
            }
        )
    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    analysis = analyze_action_trace(load_action_trace_events(log), min_rate_warn_hz=2.0)
    markdown = format_action_analysis_markdown(analysis)

    assert analysis.action_count == 4
    assert analysis.action_dim == 2
    assert analysis.zero_action_rate == 1.0
    assert analysis.repeated_action_rate == 1.0
    assert any("mostly near-zero" in warning for warning in analysis.warnings)
    assert any("mostly repeated" in warning for warning in analysis.warnings)
    assert "zero_action_rate" in markdown


def test_format_action_trace_html_contains_payload(tmp_path: Path) -> None:
    log = tmp_path / "actions.jsonl"
    log.write_text(
        json.dumps(
            {
                "header": {"stamp": {"sec": 1, "nanosec": 0}},
                "model_name": "dummy",
                "adapter_name": "DummyAdapter",
                "action_space": "eef_delta",
                "data": [0.0],
                "names": ["x"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    html = format_action_trace_html(load_action_trace_events(log), title="Trace Test")

    assert "Trace Test" in html
    assert "dummy" in html
    assert "Action Timeline" in html
