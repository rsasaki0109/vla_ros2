from __future__ import annotations

import json

from vla_zoo.runtime.dashboard import (
    dashboard_records_from_payload,
    format_comparison_dashboard_html,
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
            }
        ],
        source="results.json",
    )

    assert records[0].model_name == "dummy"
    assert records[0].source == "results.json"
    assert records[0].mean_latency_ms == 0.03


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
