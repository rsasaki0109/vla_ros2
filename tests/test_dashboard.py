from __future__ import annotations

import json
from pathlib import Path

from vla_zoo.runtime.dashboard import (
    dashboard_records_from_payload,
    format_comparison_dashboard_html,
    load_dashboard_records,
    load_runtime_dashboard_records,
)


def test_dashboard_records_from_payload_accepts_result_list() -> None:
    records = dashboard_records_from_payload(
        [
            {
                "model_name": "dummy",
                "runtime": "local",
                "ok": True,
                "adapter_queries": 2,
                "mean_latency_ms": 0.03,
                "task_success": True,
                "cube_lifted": True,
                "final_cube_distance_to_goal": 0.04,
                "phase_completion": 1.0,
            }
        ],
        source="results.json",
    )

    assert records[0].model_name == "dummy"
    assert records[0].source == "results.json"
    assert records[0].mean_latency_ms == 0.03
    assert records[0].task_success is True
    assert records[0].cube_lifted is True
    assert records[0].final_cube_distance_to_goal == 0.04
    assert records[0].phase_completion == 1.0


def test_dashboard_html_embeds_records_and_interactions() -> None:
    records = dashboard_records_from_payload(
        json.loads(
            """
            [
              {
                "model_name": "openvla",
                "runtime": "remote",
                "ok": false,
                "adapter_errors": 1,
                "last_error": "server unavailable"
              }
            ]
            """
        )
    )
    html = format_comparison_dashboard_html(records, title="Runtime Dashboard")

    assert "Runtime Dashboard" in html
    assert "openvla" in html
    assert "server unavailable" in html
    assert "filteredRecords" in html
    assert "latencyChart" in html
    assert "Fleet Health" in html
    assert "triageQueue" in html
    assert "runtimeFilter" in html
    assert "Runtime Evidence Matrix" in html
    assert "What This Does Not Claim" in html
    assert "renderEvidenceMatrix" in html
    assert "goalDistanceChart" in html
    assert "sort by task score" in html
    assert "exportCsv" in html


def test_dashboard_loads_ros_status_jsonl(tmp_path: Path) -> None:
    log = tmp_path / "vla_status.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "model_name": "dummy",
                        "ready": True,
                        "last_latency_ms": 2.0,
                        "avg_latency_ms": 2.0,
                        "status_text": "ready",
                        "metadata_json": json.dumps({"runtime": "local"}),
                    }
                ),
                json.dumps(
                    {
                        "model_name": "dummy",
                        "ready": False,
                        "last_latency_ms": 4.0,
                        "status_text": "stale image: 1.20s",
                        "metadata": {"runtime": "local"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    records = load_dashboard_records([log])

    assert len(records) == 1
    assert records[0].model_name == "dummy"
    assert records[0].runtime == "local"
    assert records[0].ok is False
    assert records[0].frames == 2
    assert records[0].adapter_queries == 2
    assert records[0].adapter_errors == 1
    assert records[0].mean_latency_ms == 3.0
    assert records[0].last_error == "stale image: 1.20s"


def test_dashboard_loads_ros_diagnostics_jsonl(tmp_path: Path) -> None:
    log = tmp_path / "diagnostics.jsonl"
    log.write_text(
        json.dumps(
            {
                "status": [
                    {
                        "name": "vla_zoo/vla_runtime_node",
                        "hardware_id": "dummy",
                        "level": 1,
                        "message": "waiting for instruction",
                        "values": [
                            {"key": "model_name", "value": "dummy"},
                            {"key": "runtime", "value": "ros2"},
                            {"key": "last_latency_ms", "value": "0.5"},
                        ],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_dashboard_records([log])

    assert len(records) == 1
    assert records[0].model_name == "dummy"
    assert records[0].runtime == "ros2"
    assert records[0].ok is False
    assert records[0].adapter_errors == 1
    assert records[0].max_latency_ms == 0.5


def test_dashboard_combines_status_and_diagnostics_logs(tmp_path: Path) -> None:
    status_log = tmp_path / "status.jsonl"
    status_log.write_text(
        json.dumps(
            {
                "model_name": "dummy",
                "ready": True,
                "last_latency_ms": 1.0,
                "status_text": "ready",
                "metadata": {"runtime": "local"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostics_log = tmp_path / "diagnostics.jsonl"
    diagnostics_log.write_text(
        json.dumps(
            {
                "status": [
                    {
                        "hardware_id": "dummy",
                        "level": 1,
                        "message": "stale image",
                        "values": [
                            {"key": "model_name", "value": "dummy"},
                            {"key": "runtime", "value": "local"},
                            {"key": "last_latency_ms", "value": "3.0"},
                        ],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_runtime_dashboard_records([status_log, diagnostics_log])

    assert len(records) == 1
    assert records[0].frames == 2
    assert records[0].adapter_queries == 2
    assert records[0].adapter_errors == 1
    assert records[0].max_latency_ms == 3.0
