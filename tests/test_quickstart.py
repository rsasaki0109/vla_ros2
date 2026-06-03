from __future__ import annotations

from pathlib import Path

from vla_zoo.demo.quickstart import (
    QUICKSTART_SCHEMA_VERSION,
    QuickstartReport,
    QuickstartRow,
    format_quickstart_html,
    format_quickstart_markdown,
    run_quickstart,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_quickstart_drives_baselines_without_heavy_deps() -> None:
    report = run_quickstart(("dummy", "scripted", "random"), episodes=3)

    assert report.ok
    assert report.episodes == 3
    by_model = {row.model: row for row in report.rows}
    assert set(by_model) == {"dummy", "scripted", "random"}
    for row in report.rows:
        assert row.error is None
        assert row.action_dim > 0
        assert row.latency_ms_p50 is not None  # the boundary actually ran
        assert len(row.sample_action) == row.action_dim


def test_run_quickstart_records_bad_adapter_as_error_row() -> None:
    report = run_quickstart(("dummy", "no-such-model"), episodes=2)

    assert report.ok is False  # one error row sinks the overall ok flag
    rows = {row.model: row for row in report.rows}
    assert rows["dummy"].ok
    assert rows["no-such-model"].error is not None
    assert rows["no-such-model"].latency_ms_p50 is None  # no fabricated latency


def test_quickstart_report_to_dict_is_schema_versioned() -> None:
    report = run_quickstart(("dummy",), episodes=2)
    payload = report.to_dict()
    assert payload["schema_version"] == QUICKSTART_SCHEMA_VERSION
    assert payload["ok"] is True
    assert payload["rows"][0]["model"] == "dummy"


def _report() -> QuickstartReport:
    row = QuickstartRow(
        model="dummy",
        action_space="eef_delta",
        action_dim=7,
        episodes=5,
        latency_ms_p50=0.01,
        latency_ms_mean=0.02,
        action_rate_hz=50000.0,
        sample_action=(0.0,) * 7,
    )
    return QuickstartReport(rows=(row,), episodes=5)


def test_quickstart_markdown_is_honest_about_baselines() -> None:
    md = format_quickstart_markdown(_report())
    assert "vla_zoo quickstart" in md
    assert "NOT VLA policies" in md
    assert "## Next steps" in md
    assert "vla_runtime_leaderboard.html" in md  # routes to real evidence


def test_quickstart_html_renders_status_and_next_steps() -> None:
    html = format_quickstart_html(_report())
    assert "<!doctype html>" in html
    assert "runtime boundary works" in html
    assert "Next: real-adapter runtime evidence" in html
    assert "vla_model_evidence_matrix.html" in html


def test_recorded_quickstart_example_exists() -> None:
    base = REPO_ROOT / "docs" / "assets" / "quickstart"
    assert (base / "report.html").is_file()
    assert (base / "report.md").is_file()
    assert (base / "report.json").is_file()
