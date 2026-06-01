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
    return (
        _DASHBOARD_HTML_TEMPLATE.replace("__VLA_ZOO_DASHBOARD_TITLE__", escape(title))
        .replace("__VLA_ZOO_DASHBOARD_DATA__", data_json)
    )


_DASHBOARD_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__VLA_ZOO_DASHBOARD_TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --panel-soft: #f8fafc;
      --text: #172033;
      --muted: #64748b;
      --line: #d8e2ee;
      --accent: #0f6f9f;
      --accent-2: #0e7490;
      --ok: #12843b;
      --ok-bg: #dcfce7;
      --error: #c51f46;
      --error-bg: #ffe4e6;
      --warn: #a16207;
      --warn-bg: #fef3c7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }
    main {
      max-width: 1360px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      margin-bottom: 20px;
    }
    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 44px);
      letter-spacing: 0;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }
    .eyebrow {
      margin-bottom: 7px;
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .header-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      min-width: 240px;
    }
    .pill, .badge {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }
    .pill {
      border: 1px solid var(--line);
      background: var(--panel);
      color: #334155;
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 150px 150px 190px auto auto auto;
      gap: 10px;
      margin: 22px 0;
      align-items: center;
    }
    input, select, button {
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      padding: 8px 10px;
      font: inherit;
    }
    button {
      cursor: pointer;
      font-weight: 750;
    }
    button:hover {
      border-color: #9fb3c8;
      background: #f8fafc;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 96px;
    }
    .card span {
      color: var(--muted);
      font-size: 13px;
    }
    .card strong {
      display: block;
      margin-top: 8px;
      font-size: 25px;
    }
    .card small {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(0, .9fr);
      gap: 16px;
    }
    .wide { grid-column: 1 / -1; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 16px;
    }
    .panel-body { padding: 14px; }
    .health-layout {
      display: grid;
      grid-template-columns: 190px 1fr;
      gap: 16px;
      align-items: center;
    }
    .score {
      display: grid;
      place-items: center;
      min-height: 150px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      text-align: center;
    }
    .score strong {
      font-size: 48px;
      line-height: 1;
    }
    .score span {
      color: var(--muted);
      font-size: 13px;
      font-weight: 750;
    }
    .health-bars { display: grid; gap: 11px; }
    .health-line {
      display: grid;
      grid-template-columns: 120px 1fr 72px;
      gap: 10px;
      align-items: center;
      color: #334155;
      font-size: 13px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }
    th {
      color: #475569;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    code {
      color: #075985;
      overflow-wrap: anywhere;
    }
    .ok { background: var(--ok-bg); color: #14532d; }
    .error { background: var(--error-bg); color: #881337; }
    .warn { background: var(--warn-bg); color: #713f12; }
    .neutral { background: #e0f2fe; color: #075985; }
    .bar-row {
      display: grid;
      grid-template-columns: 160px 1fr 82px;
      gap: 10px;
      align-items: center;
      padding: 8px 14px;
    }
    .bar {
      height: 12px;
      background: #e5e7eb;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar span {
      display: block;
      height: 100%;
      min-width: 1px;
      background: var(--accent);
    }
    .bar.error span { background: var(--error); }
    .bar.action span { background: var(--accent-2); }
    .bar.health span { background: var(--ok); }
    .bar.warn span { background: var(--warn); }
    .note {
      max-width: 520px;
      color: #334155;
    }
    .triage-list { display: grid; gap: 10px; }
    .triage-item {
      border: 1px solid var(--line);
      border-left: 4px solid var(--warn);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel-soft);
    }
    .triage-item.failed { border-left-color: var(--error); }
    .triage-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      align-items: center;
    }
    .triage-item p {
      color: #334155;
      font-size: 13px;
    }
    .empty {
      padding: 18px;
      color: var(--muted);
      background: var(--panel-soft);
      border: 1px dashed #b6c2d0;
      border-radius: 8px;
    }
    details {
      margin-top: 16px;
      padding: 12px 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    pre {
      overflow: auto;
      color: #334155;
    }
    @media (max-width: 1100px) {
      header { align-items: flex-start; flex-direction: column; }
      .header-meta { justify-content: flex-start; }
      .toolbar { grid-template-columns: 1fr 1fr; }
      .cards { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      main { padding: 24px 12px 36px; }
      .toolbar, .cards, .health-layout, .health-line, .bar-row {
        grid-template-columns: 1fr;
      }
      table { display: block; overflow-x: auto; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="eyebrow">runtime comparison artifact</div>
        <h1>__VLA_ZOO_DASHBOARD_TITLE__</h1>
        <p>
          Inspect VLA adapter readiness, latency budget, query health, and failure triage
          from static comparison JSON.
        </p>
      </div>
      <div class="header-meta">
        <span class="pill" id="recordCount">0 records</span>
        <span class="pill" id="sourceCount">0 sources</span>
        <span class="pill">offline HTML</span>
      </div>
    </header>
    <section class="toolbar">
      <input id="search" placeholder="Filter model, runtime, endpoint, note">
      <select id="status">
        <option value="all">all statuses</option>
        <option value="ok">ok only</option>
        <option value="error">errors only</option>
      </select>
      <select id="runtimeFilter">
        <option value="all">all runtimes</option>
      </select>
      <select id="sort">
        <option value="health">sort by health</option>
        <option value="latency">sort by latency</option>
        <option value="errors">sort by errors</option>
        <option value="queries">sort by queries</option>
        <option value="action">sort by action magnitude</option>
        <option value="model">sort by model</option>
      </select>
      <button id="reset">Reset</button>
      <button id="copy">Copy JSON</button>
      <button id="csv">Export CSV</button>
    </section>
    <section class="cards" id="summary"></section>
    <section class="grid">
      <div class="panel">
        <h2>Fleet Health</h2>
        <div class="panel-body" id="healthPanel"></div>
      </div>
      <div class="panel">
        <h2>Runtime Mix</h2>
        <div class="panel-body" id="runtimeMix"></div>
      </div>
      <div class="panel">
        <h2>Latency Budget</h2>
        <div id="latencyChart"></div>
      </div>
      <div class="panel">
        <h2>Adapter Error Rate</h2>
        <div id="errorChart"></div>
      </div>
      <div class="panel">
        <h2>Action Magnitude</h2>
        <div id="actionChart"></div>
      </div>
      <div class="panel">
        <h2>Triage Queue</h2>
        <div class="panel-body" id="triageQueue"></div>
      </div>
      <div class="panel wide">
        <h2>Records</h2>
        <table>
          <thead>
            <tr>
              <th>Model</th><th>Runtime</th><th>Endpoint</th><th>Status</th><th>Health</th>
              <th>Frames</th><th>Queries</th><th>Errors</th>
              <th>Mean latency</th><th>Max latency</th><th>Mean abs action</th><th>Note</th>
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
    const records = __VLA_ZOO_DASHBOARD_DATA__;

    const fmt = (value, digits = 2) =>
      value === null || value === undefined ? "-" : Number(value).toFixed(digits);
    const text = value => value === null || value === undefined ? "" : String(value);
    const esc = value => text(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
    const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

    function healthScore(record) {
      if (!record.ok) return 0;
      let score = 100;
      if (record.adapter_queries <= 0) score -= 20;
      if (record.adapter_errors > 0) score -= Math.min(55, 25 + record.adapter_errors * 10);
      if (record.frames <= 0) score -= 20;
      if (record.mean_latency_ms === null || record.mean_latency_ms === undefined) {
        score -= 15;
      } else if (record.mean_latency_ms > 250) {
        score -= 35;
      } else if (record.mean_latency_ms > 100) {
        score -= 18;
      }
      return clamp(Math.round(score), 0, 100);
    }

    function recordState(record) {
      if (!record.ok) return "failed";
      if (record.adapter_errors > 0) return "degraded";
      if ((record.mean_latency_ms ?? 0) > 250) return "slow";
      return "ready";
    }

    function stateBadge(record) {
      const state = recordState(record);
      if (state === "ready") return '<span class="badge ok">ready</span>';
      if (state === "failed") return '<span class="badge error">failed</span>';
      if (state === "slow") return '<span class="badge warn">slow</span>';
      return '<span class="badge warn">degraded</span>';
    }

    function recommendation(record) {
      const error = text(record.last_error).toLowerCase();
      if (!record.ok && error.includes("local heavy")) {
        return "Run remotely, or use --allow-local-heavy only when downloads are intended.";
      }
      if (!record.ok && error.includes("dedicated server")) {
        return "Start a model-specific remote server and compare through runtime=remote.";
      }
      if (!record.ok && error.includes("dependencies")) {
        return "Install adapter dependencies in the serving environment, not the base package.";
      }
      if (record.adapter_errors > 0) {
        return "Inspect adapter logs and response schema; keep ROS2 dry-run enabled.";
      }
      if ((record.mean_latency_ms ?? 0) > 250) {
        return "Use this as an outer-loop policy and buffer action chunks or use remote GPU.";
      }
      return "No action required.";
    }

    function populateRuntimeFilter() {
      const select = document.getElementById("runtimeFilter");
      const runtimes = [...new Set(records.map(record => text(record.runtime) || "unknown"))]
        .sort();
      select.innerHTML = '<option value="all">all runtimes</option>' + runtimes
        .map(runtime => `<option value="${esc(runtime)}">${esc(runtime)}</option>`)
        .join("");
    }

    function filteredRecords() {
      const query = document.getElementById("search").value.toLowerCase();
      const status = document.getElementById("status").value;
      const runtime = document.getElementById("runtimeFilter").value;
      const sort = document.getElementById("sort").value;
      let rows = records.filter(record => {
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
        const runtimeOk = runtime === "all" || record.runtime === runtime;
        return statusOk && runtimeOk && blob.includes(query);
      });
      rows = [...rows].sort((a, b) => {
        if (sort === "health") return healthScore(a) - healthScore(b);
        if (sort === "errors") return b.adapter_errors - a.adapter_errors;
        if (sort === "queries") return b.adapter_queries - a.adapter_queries;
        if (sort === "action") return (b.mean_abs_action ?? -1) - (a.mean_abs_action ?? -1);
        if (sort === "model") return text(a.model_name).localeCompare(text(b.model_name));
        return (b.mean_latency_ms ?? -1) - (a.mean_latency_ms ?? -1);
      });
      return rows;
    }

    function renderSummary(rows) {
      const ok = rows.filter(record => record.ok).length;
      const queries = rows.reduce((sum, record) => sum + record.adapter_queries, 0);
      const errors = rows.reduce((sum, record) => sum + record.adapter_errors, 0);
      const frames = rows.reduce((sum, record) => sum + record.frames, 0);
      const latencies = rows
        .map(record => record.mean_latency_ms)
        .filter(value => value !== null && value !== undefined);
      const avgLatency = latencies.length
        ? latencies.reduce((sum, value) => sum + value, 0) / latencies.length
        : null;
      const errorRate = queries > 0 ? errors / queries * 100 : 0;
      const health = rows.length
        ? rows.reduce((sum, record) => sum + healthScore(record), 0) / rows.length
        : 0;
      document.getElementById("summary").innerHTML = [
        ["health", `${Math.round(health)}%`, "mean readiness score"],
        ["ready", `${ok}/${rows.length}`, "passing adapters"],
        ["queries", queries, `${frames} frames observed`],
        ["error rate", `${fmt(errorRate, 1)}%`, `${errors} adapter errors`],
        ["avg latency", `${fmt(avgLatency)} ms`, "mean over available rows"],
        ["runtimes", new Set(rows.map(record => record.runtime)).size, "runtime paths"]
      ].map(([label, value, hint]) =>
        `<div class="card">
          <span>${label}</span><strong>${value}</strong><small>${hint}</small>
        </div>`
      ).join("");
    }

    function renderHealth(rows) {
      const health = rows.length
        ? rows.reduce((sum, record) => sum + healthScore(record), 0) / rows.length
        : 0;
      const counts = rows.reduce((acc, record) => {
        acc[recordState(record)] = (acc[recordState(record)] || 0) + 1;
        return acc;
      }, {});
      const lines = [
        ["ready", counts.ready || 0, "health"],
        ["degraded", counts.degraded || 0, "warn"],
        ["slow", counts.slow || 0, "warn"],
        ["failed", counts.failed || 0, "error"]
      ];
      const max = Math.max(1, ...lines.map(([, count]) => count));
      document.getElementById("healthPanel").innerHTML = `<div class="health-layout">
        <div class="score">
          <div><strong>${Math.round(health)}%</strong><span>runtime health</span></div>
        </div>
        <div class="health-bars">
          ${lines.map(([label, count, kind]) => `
            <div class="health-line">
              <span>${label}</span>
              <div class="bar ${kind}">
                <span style="width:${Math.max(2, Number(count) / max * 100)}%"></span>
              </div>
              <strong>${count}</strong>
            </div>
          `).join("")}
        </div>
      </div>`;
    }

    function renderRuntimeMix(rows) {
      const byRuntime = new Map();
      for (const record of rows) {
        const key = text(record.runtime) || "unknown";
        const entry = byRuntime.get(key) || {total: 0, ok: 0, errors: 0};
        entry.total += 1;
        entry.ok += record.ok ? 1 : 0;
        entry.errors += record.adapter_errors;
        byRuntime.set(key, entry);
      }
      const max = Math.max(1, ...[...byRuntime.values()].map(item => item.total));
      document.getElementById("runtimeMix").innerHTML = [...byRuntime.entries()]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([runtime, item]) => `
          <div class="health-line">
            <code>${esc(runtime)}</code>
            <div class="bar neutral">
              <span style="width:${Math.max(2, item.total / max * 100)}%"></span>
            </div>
            <strong>${item.ok}/${item.total}</strong>
          </div>
        `).join("") || '<div class="empty">No records match the current filters.</div>';
    }

    function renderBarChart(id, rows, metric, digits, className = "") {
      const metricValue = record => {
        if (metric === "error_rate") {
          return record.adapter_queries > 0
            ? record.adapter_errors / record.adapter_queries * 100
            : record.adapter_errors > 0 ? 100 : 0;
        }
        return Number(record[metric] ?? 0);
      };
      const values = rows.map(metricValue);
      const maxValue = Math.max(0, ...values);
      document.getElementById(id).innerHTML = rows.map(record => {
        const value = metricValue(record);
        const width = maxValue > 0 ? Math.max(2, value / maxValue * 100) : 0;
        return `<div class="bar-row">
          <code>${esc(record.model_name)}</code>
          <div class="bar ${className}"><span style="width:${width}%"></span></div>
          <strong>${fmt(value, digits)}</strong>
        </div>`;
      }).join("");
    }

    function renderTriage(rows) {
      const triage = rows
        .filter(record => recordState(record) !== "ready")
        .sort((a, b) => healthScore(a) - healthScore(b));
      document.getElementById("triageQueue").innerHTML = triage.length
        ? `<div class="triage-list">${triage.map(record => `
          <div class="triage-item ${recordState(record)}">
            <div class="triage-head">
              <strong><code>${esc(record.model_name)}</code> / ${esc(record.runtime)}</strong>
              ${stateBadge(record)}
            </div>
            <p>${esc(record.last_error || recommendation(record))}</p>
            <p><strong>Next:</strong> ${esc(recommendation(record))}</p>
          </div>
        `).join("")}</div>`
        : '<div class="empty">No failing or degraded adapters in the current view.</div>';
    }

    function renderRows(rows) {
      document.getElementById("records").innerHTML = rows.map(record => `<tr>
        <td><code>${esc(record.model_name)}</code></td>
        <td><code>${esc(record.runtime)}</code></td>
        <td>${esc(record.remote_url || "-")}</td>
        <td>${stateBadge(record)}</td>
        <td>${healthScore(record)}%</td>
        <td>${record.frames}</td>
        <td>${record.adapter_queries}</td>
        <td>${record.adapter_errors}</td>
        <td>${fmt(record.mean_latency_ms)} ms</td>
        <td>${fmt(record.max_latency_ms)} ms</td>
        <td>${fmt(record.mean_abs_action, 3)}</td>
        <td class="note">${esc(record.last_error || "-")}</td>
      </tr>`).join("");
    }

    function exportCsv(rows) {
      const headers = [
        "model_name", "runtime", "ok", "health", "remote_url", "frames",
        "adapter_queries", "adapter_errors", "mean_latency_ms", "max_latency_ms",
        "mean_abs_action", "last_error", "source"
      ];
      const quote = value => `"${text(value).replaceAll('"', '""')}"`;
      const csv = [
        headers.join(","),
        ...rows.map(record => headers.map(header => {
          if (header === "health") return healthScore(record);
          return quote(record[header]);
        }).join(","))
      ].join("\\n");
      const blob = new Blob([csv], {type: "text/csv"});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "vla_runtime_dashboard.csv";
      link.click();
      URL.revokeObjectURL(url);
    }

    async function copyJson(rows) {
      const payload = JSON.stringify(rows, null, 2);
      try {
        await navigator.clipboard.writeText(payload);
      } catch {
        const textarea = document.createElement("textarea");
        textarea.value = payload;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
    }

    function render() {
      const rows = filteredRecords();
      const sources = new Set(records.map(record => record.source).filter(Boolean));
      document.getElementById("recordCount").textContent = `${records.length} records`;
      document.getElementById("sourceCount").textContent = `${sources.size || 1} sources`;
      renderSummary(rows);
      renderHealth(rows);
      renderRuntimeMix(rows);
      renderBarChart("latencyChart", rows, "mean_latency_ms", 2);
      renderBarChart("errorChart", rows, "error_rate", 1, "error");
      renderBarChart("actionChart", rows, "mean_abs_action", 3, "action");
      renderTriage(rows);
      renderRows(rows);
      document.getElementById("raw").textContent = JSON.stringify(rows, null, 2);
    }

    populateRuntimeFilter();
    document.getElementById("search").addEventListener("input", render);
    document.getElementById("status").addEventListener("change", render);
    document.getElementById("runtimeFilter").addEventListener("change", render);
    document.getElementById("sort").addEventListener("change", render);
    document.getElementById("reset").addEventListener("click", () => {
      document.getElementById("search").value = "";
      document.getElementById("status").value = "all";
      document.getElementById("runtimeFilter").value = "all";
      document.getElementById("sort").value = "health";
      render();
    });
    document.getElementById("copy").addEventListener("click", () => copyJson(filteredRecords()));
    document.getElementById("csv").addEventListener("click", () => exportCsv(filteredRecords()));
    render();
  </script>
</body>
</html>
"""
