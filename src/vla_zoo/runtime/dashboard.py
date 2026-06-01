from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class ComparisonDashboardRecord:
    model_name: str
    runtime: str
    ok: bool
    source: str = ""
    remote_url: str | None = None
    frames: int = 0
    adapter_queries: int = 0
    adapter_errors: int = 0
    mean_latency_ms: float | None = None
    max_latency_ms: float | None = None
    mean_abs_action: float | None = None
    last_error: str | None = None


def _string(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "ok"}
    return bool(value)


def _int(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def dashboard_record_from_mapping(
    payload: Mapping[str, object],
    *,
    source: str = "",
) -> ComparisonDashboardRecord:
    return ComparisonDashboardRecord(
        model_name=_string(payload.get("model_name", payload.get("model", "unknown"))),
        runtime=_string(payload.get("runtime", "unknown")),
        ok=_bool(payload.get("ok", False)),
        source=source,
        remote_url=(
            _string(payload.get("remote_url")) if payload.get("remote_url") is not None else None
        ),
        frames=_int(payload.get("frames")),
        adapter_queries=_int(payload.get("adapter_queries")),
        adapter_errors=_int(payload.get("adapter_errors")),
        mean_latency_ms=_float(payload.get("mean_latency_ms")),
        max_latency_ms=_float(payload.get("max_latency_ms")),
        mean_abs_action=_float(payload.get("mean_abs_action")),
        last_error=(
            _string(payload.get("last_error")) if payload.get("last_error") is not None else None
        ),
    )


def dashboard_records_from_payload(
    payload: object,
    *,
    source: str = "",
) -> list[ComparisonDashboardRecord]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        payload = payload["results"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        msg = "Comparison result JSON must be a list, object, or object with a 'results' list."
        raise ValueError(msg)

    records: list[ComparisonDashboardRecord] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            msg = f"Comparison result entry {index} is not an object."
            raise ValueError(msg)
        records.append(dashboard_record_from_mapping(item, source=source))
    return records


def load_dashboard_records(paths: list[Path]) -> list[ComparisonDashboardRecord]:
    records: list[ComparisonDashboardRecord] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(dashboard_records_from_payload(payload, source=str(path)))
    return records


def format_comparison_dashboard_html(
    records: list[ComparisonDashboardRecord],
    *,
    title: str = "vla_zoo Runtime Dashboard",
) -> str:
    data_json = json.dumps([asdict(record) for record in records], indent=2).replace(
        "</",
        "<\\/",
    )
    escaped_title = escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7fafc;
      --panel: #ffffff;
      --text: #172033;
      --muted: #64748b;
      --line: #d8e0ea;
      --accent: #0369a1;
      --ok: #16a34a;
      --error: #e11d48;
      --warn: #ca8a04;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    header {{
      display: grid;
      gap: 8px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(28px, 4vw, 46px);
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) 180px 180px;
      gap: 10px;
      margin: 22px 0;
    }}
    input, select {{
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      padding: 8px 10px;
      font: inherit;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 15px;
    }}
    .card span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .card strong {{
      display: block;
      margin-top: 8px;
      font-size: 26px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .panel h2 {{
      margin: 0;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      color: #475569;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    code {{ color: #075985; }}
    .badge {{
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .ok {{ background: #bbf7d0; color: #14532d; }}
    .error {{ background: #fecdd3; color: #881337; }}
    .bar-row {{
      display: grid;
      grid-template-columns: 140px 1fr 82px;
      gap: 10px;
      align-items: center;
      padding: 8px 14px;
    }}
    .bar {{
      height: 12px;
      background: #e5e7eb;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar span {{
      display: block;
      height: 100%;
      min-width: 1px;
      background: var(--accent);
    }}
    .bar.error span {{ background: var(--error); }}
    .bar.action span {{ background: #0891b2; }}
    .note {{
      max-width: 420px;
      color: #334155;
    }}
    details {{
      margin-top: 16px;
      padding: 12px 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    pre {{
      overflow: auto;
      color: #334155;
    }}
    @media (max-width: 760px) {{
      .controls {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{escaped_title}</h1>
      <p>
        Interactive static dashboard for VLA runtime comparison results.
        Filter by model, inspect failed adapters, and compare latency/error/action metrics.
      </p>
    </header>
    <section class="controls">
      <input id="search" placeholder="Filter model, runtime, endpoint, note">
      <select id="status">
        <option value="all">all statuses</option>
        <option value="ok">ok only</option>
        <option value="error">errors only</option>
      </select>
      <select id="sort">
        <option value="latency">sort by latency</option>
        <option value="errors">sort by errors</option>
        <option value="action">sort by action magnitude</option>
        <option value="model">sort by model</option>
      </select>
    </section>
    <section class="cards" id="summary"></section>
    <section class="grid">
      <div class="panel">
        <h2>Latency</h2>
        <div id="latencyChart"></div>
      </div>
      <div class="panel">
        <h2>Errors</h2>
        <div id="errorChart"></div>
      </div>
      <div class="panel">
        <h2>Action Magnitude</h2>
        <div id="actionChart"></div>
      </div>
      <div class="panel">
        <h2>Records</h2>
        <table>
          <thead>
            <tr>
              <th>Model</th><th>Runtime</th><th>Endpoint</th><th>Status</th>
              <th>Frames</th><th>Queries</th><th>Errors</th>
              <th>Mean latency</th><th>Mean abs action</th><th>Note</th>
            </tr>
          </thead>
          <tbody id="records"></tbody>
        </table>
      </div>
    </section>
    <details>
      <summary>Raw JSON</summary>
      <pre id="raw"></pre>
    </details>
  </main>
  <script>
    const records = {data_json};

    const fmt = (value, digits = 2) =>
      value === null || value === undefined ? "-" : Number(value).toFixed(digits);
    const text = value => value === null || value === undefined ? "" : String(value);
    const esc = value => text(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");

    function filteredRecords() {{
      const query = document.getElementById("search").value.toLowerCase();
      const status = document.getElementById("status").value;
      const sort = document.getElementById("sort").value;
      let rows = records.filter(record => {{
        const blob = [
          record.model_name,
          record.runtime,
          record.remote_url,
          record.last_error,
          record.source
        ].join(" ").toLowerCase();
        const statusOk = status === "all" ||
          (status === "ok" && record.ok) ||
          (status === "error" && !record.ok);
        return statusOk && blob.includes(query);
      }});
      rows = [...rows].sort((a, b) => {{
        if (sort === "errors") return b.adapter_errors - a.adapter_errors;
        if (sort === "action") {{
          return (b.mean_abs_action ?? -1) - (a.mean_abs_action ?? -1);
        }}
        if (sort === "model") return text(a.model_name).localeCompare(text(b.model_name));
        return (b.mean_latency_ms ?? -1) - (a.mean_latency_ms ?? -1);
      }});
      return rows;
    }}

    function renderSummary(rows) {{
      const ok = rows.filter(record => record.ok).length;
      const queries = rows.reduce((sum, record) => sum + record.adapter_queries, 0);
      const errors = rows.reduce((sum, record) => sum + record.adapter_errors, 0);
      const latencies = rows
        .map(record => record.mean_latency_ms)
        .filter(value => value !== null && value !== undefined);
      const avgLatency = latencies.length
        ? latencies.reduce((sum, value) => sum + value, 0) / latencies.length
        : null;
      document.getElementById("summary").innerHTML = [
        ["models", rows.length],
        ["ok", ok],
        ["adapter queries", queries],
        ["adapter errors", errors],
        ["avg latency ms", fmt(avgLatency)]
      ].map(([label, value]) =>
        `<div class="card"><span>${{label}}</span><strong>${{value}}</strong></div>`
      ).join("");
    }}

    function renderBarChart(id, rows, metric, digits, className = "") {{
      const values = rows.map(record => Number(record[metric] ?? 0));
      const maxValue = Math.max(0, ...values);
      document.getElementById(id).innerHTML = rows.map(record => {{
        const value = Number(record[metric] ?? 0);
        const width = maxValue > 0 ? Math.max(2, value / maxValue * 100) : 0;
        return `<div class="bar-row">
          <code>${{esc(record.model_name)}}</code>
          <div class="bar ${{className}}"><span style="width:${{width}}%"></span></div>
          <strong>${{fmt(value, digits)}}</strong>
        </div>`;
      }}).join("");
    }}

    function renderRows(rows) {{
      document.getElementById("records").innerHTML = rows.map(record => {{
        const badge = record.ok
          ? '<span class="badge ok">ok</span>'
          : '<span class="badge error">error</span>';
        return `<tr>
          <td><code>${{esc(record.model_name)}}</code></td>
          <td><code>${{esc(record.runtime)}}</code></td>
          <td>${{esc(record.remote_url || "-")}}</td>
          <td>${{badge}}</td>
          <td>${{record.frames}}</td>
          <td>${{record.adapter_queries}}</td>
          <td>${{record.adapter_errors}}</td>
          <td>${{fmt(record.mean_latency_ms)}} ms</td>
          <td>${{fmt(record.mean_abs_action, 3)}}</td>
          <td class="note">${{esc(record.last_error || "-")}}</td>
        </tr>`;
      }}).join("");
    }}

    function render() {{
      const rows = filteredRecords();
      renderSummary(rows);
      renderBarChart("latencyChart", rows, "mean_latency_ms", 2);
      renderBarChart("errorChart", rows, "adapter_errors", 0, "error");
      renderBarChart("actionChart", rows, "mean_abs_action", 3, "action");
      renderRows(rows);
      document.getElementById("raw").textContent = JSON.stringify(rows, null, 2);
    }}

    document.getElementById("search").addEventListener("input", render);
    document.getElementById("status").addEventListener("change", render);
    document.getElementById("sort").addEventListener("change", render);
    render();
  </script>
</body>
</html>
"""
