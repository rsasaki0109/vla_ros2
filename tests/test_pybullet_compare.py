from __future__ import annotations

from vla_zoo.demo.pybullet import (
    PyBulletComparisonResult,
    format_pybullet_comparison_html,
    format_pybullet_comparison_markdown,
)


def test_pybullet_comparison_markdown_includes_remote_url() -> None:
    markdown = format_pybullet_comparison_markdown(
        [
            PyBulletComparisonResult(
                model_name="openvla",
                runtime="remote",
                ok=True,
                remote_url="http://gpu-box:8001",
                frames=10,
                adapter_queries=2,
                mean_latency_ms=12.34,
                mean_abs_action=0.25,
            )
        ]
    )

    assert "`openvla`" in markdown
    assert "http://gpu-box:8001" in markdown
    assert "12.34" in markdown


def test_pybullet_comparison_html_includes_summary_and_json() -> None:
    html = format_pybullet_comparison_html(
        [
            PyBulletComparisonResult(
                model_name="dummy",
                runtime="local",
                ok=True,
                frames=14,
                adapter_queries=2,
                mean_latency_ms=0.03,
                mean_abs_action=0.0,
            )
        ],
        title="Dummy Report",
    )

    assert "<!doctype html>" in html
    assert "Dummy Report" in html
    assert "<code>dummy</code>" in html
    assert "Raw JSON" in html
