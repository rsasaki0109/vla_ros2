from __future__ import annotations

import pytest

from vla_zoo.benchmark.aggregate import (
    AGGREGATE_SCHEMA_VERSION,
    format_aggregate_markdown,
    rank_summaries,
)
from vla_zoo.benchmark.results import BenchmarkSummary


def _summary(
    model: str,
    *,
    p50: float | None,
    rate: float | None = 1.0,
    success_rate: float | None = None,
) -> BenchmarkSummary:
    return BenchmarkSummary(
        model=model,
        source="pybullet-action-probe",
        sample_count=21,
        success_count=0,
        success_rate=success_rate,
        latency_ms_p50=p50,
        latency_ms_p95=None if p50 is None else p50 * 1.5,
        latency_ms_mean=p50,
        action_rate_hz=rate,
        exception_count=0,
    )


def test_rank_by_latency_p50_lower_is_better() -> None:
    report = rank_summaries(
        [_summary("openvla", p50=2000.0), _summary("smolvla", p50=382.0)],
        metric="latency_ms_p50",
    )
    assert report.metric == "latency_ms_p50"
    assert report.lower_is_better is True
    assert report.count == 2
    assert [(e.rank, e.summary.model) for e in report.ranked] == [
        (1, "smolvla"),
        (2, "openvla"),
    ]


def test_rank_by_action_rate_higher_is_better() -> None:
    report = rank_summaries(
        [_summary("slow", p50=100.0, rate=2.0), _summary("fast", p50=100.0, rate=9.0)],
        metric="action_rate_hz",
    )
    assert report.lower_is_better is False
    assert [e.summary.model for e in report.ranked] == ["fast", "slow"]


def test_ties_share_a_rank() -> None:
    report = rank_summaries(
        [
            _summary("a", p50=100.0),
            _summary("b", p50=100.0),
            _summary("c", p50=200.0),
        ],
        metric="latency_ms_p50",
    )
    ranks = {e.summary.model: e.rank for e in report.ranked}
    assert ranks["a"] == 1
    assert ranks["b"] == 1  # tie shares rank
    assert ranks["c"] == 3  # competition ranking skips 2


def test_missing_metric_is_unranked_and_last() -> None:
    report = rank_summaries(
        [_summary("has", p50=500.0), _summary("missing", p50=None)],
        metric="latency_ms_p50",
    )
    last = report.ranked[-1]
    assert last.summary.model == "missing"
    assert last.rank is None
    assert last.metric_value is None
    assert report.ranked[0].rank == 1


def test_unsupported_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        rank_summaries([_summary("a", p50=1.0)], metric="bogus")


def test_aggregate_to_dict_is_schema_versioned() -> None:
    report = rank_summaries([_summary("a", p50=1.0)], metric="latency_ms_p50")
    payload = report.to_dict()
    assert payload["schema_version"] == AGGREGATE_SCHEMA_VERSION
    assert payload["metric"] == "latency_ms_p50"
    assert payload["count"] == 1
    assert payload["ranked"][0]["rank"] == 1
    assert payload["ranked"][0]["summary"]["model"] == "a"


def test_markdown_is_ranked_and_runtime_centric() -> None:
    report = rank_summaries(
        [_summary("smolvla", p50=382.0), _summary("openvla", p50=2000.0)],
        metric="latency_ms_p50",
    )
    md = format_aggregate_markdown(report)
    assert "Ranked by: `latency_ms_p50`" in md
    assert "| Rank |" in md
    assert "not by robot task-success quality" in md
    # success rate stays blank for a no-task-success run
    assert "| - |" in md


def test_per_model_rollup_across_runs() -> None:
    # three smolvla runs + one openvla run -> per-model best/median/worst
    report = rank_summaries(
        [
            _summary("smolvla", p50=300.0),
            _summary("smolvla", p50=500.0),
            _summary("smolvla", p50=400.0),
            _summary("openvla", p50=2000.0),
        ],
        metric="latency_ms_p50",
    )
    groups = {g.model: g for g in report.groups}
    assert groups["smolvla"].run_count == 3
    assert groups["smolvla"].best == 300.0  # lowest latency is best
    assert groups["smolvla"].worst == 500.0
    assert groups["smolvla"].median == 400.0
    assert groups["openvla"].run_count == 1
    # groups are ordered best-first in the metric direction
    assert [g.model for g in report.groups] == ["smolvla", "openvla"]


def test_per_model_rollup_best_follows_action_rate_direction() -> None:
    report = rank_summaries(
        [_summary("m", p50=100.0, rate=2.0), _summary("m", p50=100.0, rate=9.0)],
        metric="action_rate_hz",
    )
    group = report.groups[0]
    assert group.best == 9.0  # higher action rate is best
    assert group.worst == 2.0
    assert group.median == 5.5


def test_rollup_with_no_metric_values_is_none_and_last() -> None:
    report = rank_summaries(
        [_summary("has", p50=100.0), _summary("none", p50=None)],
        metric="latency_ms_p50",
    )
    groups = {g.model: g for g in report.groups}
    assert groups["none"].best is None
    assert groups["none"].median is None
    assert report.groups[-1].model == "none"  # metric-less model ordered last


def test_markdown_has_per_model_rollup_section() -> None:
    report = rank_summaries(
        [_summary("smolvla", p50=300.0), _summary("smolvla", p50=500.0)],
        metric="latency_ms_p50",
    )
    md = format_aggregate_markdown(report)
    assert "## Per-model roll-up" in md
    assert "| Model | Runs | Best | Median | Worst |" in md


def test_groups_in_aggregate_to_dict() -> None:
    report = rank_summaries([_summary("a", p50=1.0)], metric="latency_ms_p50")
    payload = report.to_dict()
    assert payload["groups"][0]["model"] == "a"
    assert payload["groups"][0]["run_count"] == 1
