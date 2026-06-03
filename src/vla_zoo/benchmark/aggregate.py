"""Aggregate multiple ``vla-zoo-benchmark/v1`` summaries into one ranked table.

``bench-report`` lists summaries in input order; this module *ranks* them by a
chosen runtime metric (latency p50/p95/mean ascending, action rate descending) and
emits both a Markdown table and a machine-readable aggregate JSON. It is
runtime-centric: ranking is by latency / action throughput, never by task success.
Summaries missing the chosen metric are listed unranked at the end rather than
silently dropped or treated as best/worst.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from vla_zoo.benchmark.results import RESULT_SCHEMA_VERSION, BenchmarkSummary

#: Schema identifier for the aggregate artifact (distinct from the per-run schema).
AGGREGATE_SCHEMA_VERSION = "vla-zoo-benchmark-aggregate/v1"

#: Metrics that can be ranked, mapped to whether lower is better (ascending rank).
#: Latencies: lower is better. Action rate: higher is better.
RANKABLE_METRICS: dict[str, bool] = {
    "latency_ms_p50": True,
    "latency_ms_p95": True,
    "latency_ms_mean": True,
    "action_rate_hz": False,
}

DEFAULT_METRIC = "latency_ms_p50"

_DISCLAIMER = (
    "Runtime-centric aggregate. Rank is by the selected latency / action-rate metric, "
    "not by robot task-success quality. A blank success rate means the source made no "
    "task-success claim (for example, replayed action logs)."
)


@dataclass(frozen=True)
class RankedSummary:
    """One summary with its rank under the chosen metric.

    ``rank`` is ``None`` when the summary does not carry the ranking metric; such
    rows are reported after the ranked ones. Ties share a rank (competition
    ranking: 1, 2, 2, 4).
    """

    rank: int | None
    metric: str
    metric_value: float | None
    summary: BenchmarkSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "metric": self.metric,
            "metric_value": self.metric_value,
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True)
class ModelRollup:
    """Best / worst / median of the ranking metric across one model's runs.

    "Best" and "worst" follow the metric direction (for latency, best is the lowest;
    for action rate, best is the highest). Values are ``None`` when none of the
    model's runs carry the metric. This turns a flat ranked table into a per-model
    stability view when several runs of the same model are aggregated.
    """

    model: str
    run_count: int
    metric: str
    best: float | None
    worst: float | None
    median: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "run_count": self.run_count,
            "metric": self.metric,
            "best": self.best,
            "worst": self.worst,
            "median": self.median,
        }


@dataclass(frozen=True)
class AggregateReport:
    """A ranked aggregate over several benchmark summaries."""

    metric: str
    lower_is_better: bool
    ranked: tuple[RankedSummary, ...]
    groups: tuple[ModelRollup, ...] = ()
    schema_version: str = AGGREGATE_SCHEMA_VERSION

    @property
    def count(self) -> int:
        return len(self.ranked)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "lower_is_better": self.lower_is_better,
            "count": self.count,
            "source_schema_version": RESULT_SCHEMA_VERSION,
            "ranked": [entry.to_dict() for entry in self.ranked],
            "groups": [group.to_dict() for group in self.groups],
        }


def rank_summaries(
    summaries: Sequence[BenchmarkSummary],
    *,
    metric: str = DEFAULT_METRIC,
) -> AggregateReport:
    """Rank ``summaries`` by ``metric`` and return an :class:`AggregateReport`.

    Summaries with a ``None`` value for the metric are placed after the ranked rows
    (in their original order) with ``rank=None``. Ties share a rank.
    """

    if metric not in RANKABLE_METRICS:
        allowed = ", ".join(sorted(RANKABLE_METRICS))
        msg = f"Unsupported metric {metric!r}; expected one of: {allowed}"
        raise ValueError(msg)

    lower_is_better = RANKABLE_METRICS[metric]

    def _value(summary: BenchmarkSummary) -> float | None:
        raw = getattr(summary, metric)
        return None if raw is None else float(raw)

    with_value = [s for s in summaries if _value(s) is not None]
    without_value = [s for s in summaries if _value(s) is None]

    ordered = sorted(
        with_value,
        key=lambda s: _value(s),  # type: ignore[arg-type, return-value]
        reverse=not lower_is_better,
    )

    ranked: list[RankedSummary] = []
    previous_value: float | None = None
    previous_rank = 0
    for position, summary in enumerate(ordered, start=1):
        value = _value(summary)
        # A tie shares the prior rank (competition ranking: 1, 2, 2, 4).
        is_tie = previous_value is not None and value == previous_value
        rank = previous_rank if is_tie else position
        ranked.append(
            RankedSummary(rank=rank, metric=metric, metric_value=value, summary=summary)
        )
        previous_value = value
        previous_rank = rank

    for summary in without_value:
        ranked.append(
            RankedSummary(rank=None, metric=metric, metric_value=None, summary=summary)
        )

    groups = _rollup_by_model(summaries, metric=metric, lower_is_better=lower_is_better)

    return AggregateReport(
        metric=metric,
        lower_is_better=lower_is_better,
        ranked=tuple(ranked),
        groups=groups,
    )


def _rollup_by_model(
    summaries: Sequence[BenchmarkSummary],
    *,
    metric: str,
    lower_is_better: bool,
) -> tuple[ModelRollup, ...]:
    """Group summaries by model and roll up the metric across each model's runs.

    Models are ordered by their best value in the ranking direction; a model whose
    runs all lack the metric is placed last (in first-seen order). Insertion order is
    preserved within ties so the output is deterministic.
    """

    order: list[str] = []
    values_by_model: dict[str, list[float]] = {}
    runs_by_model: dict[str, int] = {}
    for summary in summaries:
        model = summary.model
        if model not in runs_by_model:
            order.append(model)
            values_by_model[model] = []
            runs_by_model[model] = 0
        runs_by_model[model] += 1
        raw = getattr(summary, metric)
        if raw is not None:
            values_by_model[model].append(float(raw))

    rollups: list[ModelRollup] = []
    for model in order:
        values = values_by_model[model]
        if values:
            best = min(values) if lower_is_better else max(values)
            worst = max(values) if lower_is_better else min(values)
            rollups.append(
                ModelRollup(
                    model=model,
                    run_count=runs_by_model[model],
                    metric=metric,
                    best=best,
                    worst=worst,
                    median=float(median(values)),
                )
            )
        else:
            rollups.append(
                ModelRollup(
                    model=model,
                    run_count=runs_by_model[model],
                    metric=metric,
                    best=None,
                    worst=None,
                    median=None,
                )
            )

    def _sort_key(rollup: ModelRollup) -> tuple[int, float]:
        if rollup.best is None:
            return (1, 0.0)
        return (0, rollup.best if lower_is_better else -rollup.best)

    return tuple(sorted(rollups, key=_sort_key))


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _success_text(summary: BenchmarkSummary) -> str:
    return "-" if summary.success_rate is None else f"{summary.success_rate:.2%}"


def format_aggregate_markdown(
    report: AggregateReport,
    *,
    title: str = "Benchmark Aggregate (Ranked)",
) -> str:
    """Render a ranked aggregate as a Markdown table."""

    direction = "lower is better" if report.lower_is_better else "higher is better"
    headers = [
        "Rank",
        "Model",
        "Source",
        "Samples",
        "Success rate",
        "Latency p50 (ms)",
        "Latency p95 (ms)",
        "Latency mean (ms)",
        "Action rate (Hz)",
        "Exceptions",
    ]
    lines = [
        f"# {title}",
        "",
        f"- Schema: `{report.schema_version}`",
        f"- Ranked by: `{report.metric}` ({direction})",
        f"- Entries: {report.count}",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for entry in report.ranked:
        summary = entry.summary
        rank = "-" if entry.rank is None else str(entry.rank)
        cells = [
            rank,
            summary.model or "-",
            summary.source or "-",
            str(summary.sample_count),
            _success_text(summary),
            _num(summary.latency_ms_p50),
            _num(summary.latency_ms_p95),
            _num(summary.latency_ms_mean),
            _num(summary.action_rate_hz),
            str(summary.exception_count),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    if report.groups:
        direction = "lowest" if report.lower_is_better else "highest"
        group_headers = ["Model", "Runs", "Best", "Median", "Worst"]
        lines.extend(
            [
                "",
                "## Per-model roll-up",
                "",
                f"Best is the {direction} `{report.metric}` across each model's runs.",
                "",
                "| " + " | ".join(group_headers) + " |",
                "|" + "|".join("---" for _ in group_headers) + "|",
            ]
        )
        for group in report.groups:
            cells = [
                group.model or "-",
                str(group.run_count),
                _num(group.best),
                _num(group.median),
                _num(group.worst),
            ]
            lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", _DISCLAIMER, ""])
    return "\n".join(lines) + "\n"
