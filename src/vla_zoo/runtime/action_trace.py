from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class ActionTraceEvent:
    timestamp_sec: float
    relative_sec: float
    model_name: str
    adapter_name: str
    action_space: str
    data: list[float]
    names: list[str]
    confidence: float | None = None
    dt: float | None = None
    chunk_index: int = 0
    metadata: dict[str, object] | None = None

    @property
    def magnitude(self) -> float:
        return sum(abs(value) for value in self.data)


@dataclass(frozen=True)
class ActionTraceSummary:
    action_count: int
    duration_sec: float
    model_names: list[str]
    action_spaces: list[str]
    mean_magnitude: float | None
    max_magnitude: float | None


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def _timestamp_sec(payload: dict[str, object]) -> float:
    header = payload.get("header")
    if not isinstance(header, dict):
        return 0.0
    stamp = header.get("stamp")
    if not isinstance(stamp, dict):
        return 0.0
    sec = stamp.get("sec", 0)
    nanosec = stamp.get("nanosec", 0)
    return float(sec) + float(nanosec) / 1_000_000_000.0


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    values: list[float] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int | float):
            values.append(float(item))
    return values


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _metadata(payload: dict[str, object]) -> dict[str, object] | None:
    raw = payload.get("metadata")
    if isinstance(raw, dict):
        return dict(raw)
    raw_text = payload.get("metadata_json")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"metadata_json": raw_text}
    return parsed if isinstance(parsed, dict) else None


def load_action_trace_events(action_log: Path) -> list[ActionTraceEvent]:
    rows = _read_jsonl(action_log)
    if not rows:
        return []
    base_time = min(_timestamp_sec(row) for row in rows)
    events: list[ActionTraceEvent] = []
    for row in rows:
        timestamp = _timestamp_sec(row)
        confidence = row.get("confidence")
        dt = row.get("dt")
        chunk_index = row.get("chunk_index")
        events.append(
            ActionTraceEvent(
                timestamp_sec=timestamp,
                relative_sec=max(0.0, timestamp - base_time),
                model_name=str(row.get("model_name", "unknown")),
                adapter_name=str(row.get("adapter_name", "")),
                action_space=str(row.get("action_space", "unknown")),
                data=_float_list(row.get("data")),
                names=_string_list(row.get("names")),
                confidence=float(confidence) if isinstance(confidence, int | float) else None,
                dt=float(dt) if isinstance(dt, int | float) else None,
                chunk_index=int(chunk_index) if isinstance(chunk_index, int) else 0,
                metadata=_metadata(row),
            )
        )
    return sorted(events, key=lambda event: event.timestamp_sec)


def summarize_action_trace(events: list[ActionTraceEvent]) -> ActionTraceSummary:
    if not events:
        return ActionTraceSummary(
            action_count=0,
            duration_sec=0.0,
            model_names=[],
            action_spaces=[],
            mean_magnitude=None,
            max_magnitude=None,
        )
    magnitudes = [event.magnitude for event in events]
    return ActionTraceSummary(
        action_count=len(events),
        duration_sec=max(event.relative_sec for event in events),
        model_names=sorted({event.model_name for event in events}),
        action_spaces=sorted({event.action_space for event in events}),
        mean_magnitude=mean(magnitudes),
        max_magnitude=max(magnitudes),
    )


def format_action_trace_html(
    events: list[ActionTraceEvent],
    *,
    title: str = "vla_zoo Action Trace",
) -> str:
    summary = summarize_action_trace(events)
    payload = {
        "summary": asdict(summary),
        "events": [asdict(event) for event in events],
    }
    data_json = json.dumps(payload, indent=2).replace("</", "<\\/")
    return (
        _ACTION_TRACE_HTML_TEMPLATE.replace("__VLA_ZOO_ACTION_TRACE_TITLE__", escape(title))
        .replace("__VLA_ZOO_ACTION_TRACE_DATA__", data_json)
    )


_ACTION_TRACE_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__VLA_ZOO_ACTION_TRACE_TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #64748b;
      --line: #d8e2ee;
      --accent: #0f6f9f;
      --green: #12843b;
      --red: #c51f46;
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
      max-width: 1280px;
      margin: 0 auto;
      padding: 30px 18px 46px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 44px);
      letter-spacing: 0;
    }
    p { margin: 7px 0 0; color: var(--muted); line-height: 1.55; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      padding: 4px 10px;
      color: #334155;
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .card {
      min-height: 92px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 13px;
    }
    .card span { color: var(--muted); font-size: 13px; }
    .card strong { display: block; margin-top: 8px; font-size: 24px; }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
      margin-bottom: 16px;
    }
    .panel h2 {
      margin: 0;
      border-bottom: 1px solid var(--line);
      padding: 13px 15px;
      font-size: 16px;
    }
    .panel-body { padding: 14px; }
    canvas {
      display: block;
      width: 100%;
      height: 320px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
    }
    .controls {
      display: grid;
      grid-template-columns: 1fr 160px;
      gap: 12px;
      align-items: center;
      margin-top: 12px;
    }
    input[type="range"] { width: 100%; }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }
    th { color: #475569; font-size: 12px; text-transform: uppercase; }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
        "Liberation Mono", monospace;
      font-size: 12px;
    }
    .detail {
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 10px;
      color: #334155;
      font-size: 14px;
    }
    .detail strong { color: var(--text); }
    @media (max-width: 860px) {
      header { display: block; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .controls, .detail { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>__VLA_ZOO_ACTION_TRACE_TITLE__</h1>
        <p>Static replay of VLA action messages recorded from ROS2.</p>
      </div>
      <span class="pill" id="model-pill">no model</span>
    </header>

    <section class="cards">
      <div class="card"><span>Actions</span><strong id="count">0</strong></div>
      <div class="card"><span>Duration</span><strong id="duration">0.00s</strong></div>
      <div class="card"><span>Mean |action|</span><strong id="mean-mag">-</strong></div>
      <div class="card"><span>Max |action|</span><strong id="max-mag">-</strong></div>
      <div class="card"><span>Action Space</span><strong id="space">-</strong></div>
    </section>

    <section class="panel">
      <h2>Action Timeline</h2>
      <div class="panel-body">
        <canvas id="chart" width="1200" height="360"></canvas>
        <div class="controls">
          <input id="scrubber" type="range" min="0" max="0" value="0">
          <span class="pill" id="scrub-label">event 0 / 0</span>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Selected Action</h2>
      <div class="panel-body detail" id="detail"></div>
    </section>

    <section class="panel">
      <h2>Recent Events</h2>
      <div class="panel-body">
        <table>
          <thead>
            <tr>
              <th>t</th>
              <th>model</th>
              <th>space</th>
              <th>|action|</th>
              <th>data</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const trace = __VLA_ZOO_ACTION_TRACE_DATA__;
    const events = trace.events || [];
    const summary = trace.summary || {};
    const fmt = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : "-";

    document.getElementById("count").textContent = summary.action_count ?? 0;
    document.getElementById("duration").textContent = `${fmt(summary.duration_sec || 0)}s`;
    document.getElementById("mean-mag").textContent = fmt(summary.mean_magnitude);
    document.getElementById("max-mag").textContent = fmt(summary.max_magnitude);
    document.getElementById("space").textContent = (summary.action_spaces || []).join(", ") || "-";
    document.getElementById("model-pill").textContent =
      (summary.model_names || []).join(", ") || "no model";

    const scrubber = document.getElementById("scrubber");
    scrubber.max = Math.max(0, events.length - 1);

    function magnitude(event) {
      return (event.data || []).reduce((sum, value) => sum + Math.abs(Number(value) || 0), 0);
    }

    function draw(selectedIndex = 0) {
      const canvas = document.getElementById("chart");
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      const pad = 42;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#fbfdff";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#d8e2ee";
      ctx.lineWidth = 1;
      for (let i = 0; i < 5; i += 1) {
        const y = pad + i * ((height - pad * 2) / 4);
        ctx.beginPath();
        ctx.moveTo(pad, y);
        ctx.lineTo(width - pad, y);
        ctx.stroke();
      }
      if (!events.length) {
        ctx.fillStyle = "#64748b";
        ctx.font = "18px sans-serif";
        ctx.fillText("No action records", pad, height / 2);
        return;
      }
      const maxT = Math.max(...events.map((event) => event.relative_sec || 0), 1);
      const mags = events.map(magnitude);
      const maxM = Math.max(...mags, 1e-6);
      ctx.strokeStyle = "#0f6f9f";
      ctx.lineWidth = 2;
      ctx.beginPath();
      events.forEach((event, index) => {
        const x = pad + ((event.relative_sec || 0) / maxT) * (width - pad * 2);
        const y = height - pad - (mags[index] / maxM) * (height - pad * 2);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      events.forEach((event, index) => {
        const x = pad + ((event.relative_sec || 0) / maxT) * (width - pad * 2);
        const y = height - pad - (mags[index] / maxM) * (height - pad * 2);
        ctx.fillStyle = index === selectedIndex ? "#c51f46" : "#12843b";
        ctx.beginPath();
        ctx.arc(x, y, index === selectedIndex ? 6 : 3, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.fillStyle = "#64748b";
      ctx.font = "13px sans-serif";
      ctx.fillText("time", width - pad - 24, height - 12);
      ctx.fillText("|action|", 10, pad - 14);
    }

    function showDetail(index) {
      const event = events[index];
      document.getElementById("scrub-label").textContent =
        events.length ? `event ${index + 1} / ${events.length}` : "event 0 / 0";
      if (!event) {
        document.getElementById("detail").innerHTML = "<p>No event selected.</p>";
        return;
      }
      const named = (event.data || []).map((value, itemIndex) => {
        const name = (event.names || [])[itemIndex] || `a${itemIndex}`;
        return `${name}: ${Number(value).toFixed(4)}`;
      });
      document.getElementById("detail").innerHTML = `
        <div>Time</div><strong>${fmt(event.relative_sec, 3)}s</strong>
        <div>Model</div><strong>${event.model_name} / ${event.adapter_name || "-"}</strong>
        <div>Action space</div><strong>${event.action_space}</strong>
        <div>Magnitude</div><strong>${fmt(magnitude(event), 4)}</strong>
        <div>Confidence</div><strong>${fmt(event.confidence, 3)}</strong>
        <div>Data</div><code>${named.join(" | ")}</code>
      `;
    }

    function renderRows() {
      const rows = document.getElementById("rows");
      rows.innerHTML = events.slice(-80).reverse().map((event) => `
        <tr>
          <td>${fmt(event.relative_sec, 3)}s</td>
          <td>${event.model_name}</td>
          <td>${event.action_space}</td>
          <td>${fmt(magnitude(event), 4)}</td>
          <td><code>${
            (event.data || []).map((value) => Number(value).toFixed(4)).join(", ")
          }</code></td>
        </tr>
      `).join("");
    }

    scrubber.addEventListener("input", () => {
      const index = Number(scrubber.value);
      draw(index);
      showDetail(index);
    });
    draw(0);
    showDetail(0);
    renderRows();
  </script>
</body>
</html>
"""
