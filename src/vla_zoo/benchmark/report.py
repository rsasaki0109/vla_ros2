"""Render benchmark summaries into the report/dashboard surface.

This turns one or more :class:`~vla_zoo.benchmark.results.BenchmarkSummary` objects into
a comparison table (Markdown and standalone HTML) so latency / action-rate results sit
alongside the evidence matrix and artifact index on the Pages surface. The report is
runtime-centric: a blank success rate means the source made no task-success claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from vla_zoo.benchmark.results import BenchmarkSummary

_DISCLAIMER = (
    "Runtime-centric benchmark comparison. It measures latency and action throughput, "
    "not robot task-success quality. A blank success rate means the source made no "
    "task-success claim (for example, replayed action logs)."
)

_COLUMNS = (
    ("Model", "model"),
    ("Source", "source"),
    ("Samples", "sample_count"),
    ("Success rate", "success_rate"),
    ("Latency p50 (ms)", "latency_ms_p50"),
    ("Latency p95 (ms)", "latency_ms_p95"),
    ("Latency mean (ms)", "latency_ms_mean"),
    ("Action rate (Hz)", "action_rate_hz"),
    ("Exceptions", "exception_count"),
)


def _success_text(summary: BenchmarkSummary) -> str:
    return "-" if summary.success_rate is None else f"{summary.success_rate:.2%}"


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _row_cells(summary: BenchmarkSummary) -> list[str]:
    return [
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


def format_benchmark_report_markdown(
    summaries: Sequence[BenchmarkSummary],
    *,
    title: str = "Benchmark Comparison",
) -> str:
    """Render a multi-summary comparison table as Markdown."""

    headers = [label for label, _ in _COLUMNS]
    lines = [
        f"# {title}",
        "",
        f"- Schema: `{summaries[0].schema_version if summaries else 'vla-zoo-benchmark/v1'}`",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(_row_cells(summary)) + " |")
    lines.extend(["", _DISCLAIMER, ""])
    return "\n".join(lines) + "\n"


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7fb; --panel: #ffffff; --text: #172033; --muted: #64748b;
      --line: #d8e2ee; --accent: #0f6f9f; --head: #eef4fb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    main { max-width: 1100px; margin: 0 auto; padding: 32px 20px 48px; }
    h1 { margin: 0 0 4px; font-size: clamp(26px, 4vw, 40px); }
    p { color: var(--muted); line-height: 1.55; }
    .panel {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 14px; padding: 18px 20px; margin-top: 18px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td {
      text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }
    th { background: var(--head); color: var(--accent); font-weight: 600; }
    td:first-child, th:first-child { font-weight: 600; }
    tbody tr:last-child td { border-bottom: none; }
    .note { margin-top: 16px; font-size: 13px; }
    code { background: #eef2f7; padding: 1px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <main>
    <h1>__TITLE__</h1>
    <p>Schema <code>__SCHEMA__</code></p>
    <div class="panel">
      <table>
        <thead><tr>__HEAD__</tr></thead>
        <tbody>
__ROWS__
        </tbody>
      </table>
      <p class="note">__DISCLAIMER__</p>
    </div>
  </main>
</body>
</html>
"""


def format_benchmark_report_html(
    summaries: Sequence[BenchmarkSummary],
    *,
    title: str = "Benchmark Comparison",
) -> str:
    """Render a multi-summary comparison table as standalone HTML."""

    schema = summaries[0].schema_version if summaries else "vla-zoo-benchmark/v1"
    head = "".join(f"<th>{escape(label)}</th>" for label, _ in _COLUMNS)
    rows = []
    for summary in summaries:
        cells = "".join(f"<td>{escape(cell)}</td>" for cell in _row_cells(summary))
        rows.append(f"          <tr>{cells}</tr>")
    return (
        _HTML_TEMPLATE.replace("__TITLE__", escape(title))
        .replace("__SCHEMA__", escape(schema))
        .replace("__HEAD__", head)
        .replace("__ROWS__", "\n".join(rows))
        .replace("__DISCLAIMER__", escape(_DISCLAIMER))
    )
