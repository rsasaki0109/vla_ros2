from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

from vla_zoo.demo.gif_suite import PyBulletGifSpec
from vla_zoo.demo.pybullet import (
    HEAVY_LOCAL_MODELS,
    RenderSample,
    run_simulation,
    summarize_pybullet_samples,
)

ACTION_PLAYGROUND_SCHEMA = "vla_zoo.action_playground.v1"
Simulator = Callable[[PyBulletGifSpec], list[RenderSample]]


@dataclass(frozen=True)
class ActionPlaygroundFrame:
    frame_index: int
    phase: str
    sim_time: float
    scripted_action: tuple[float, float, float, float]
    adapter_action: tuple[float, float, float, float] | None
    displayed_action: tuple[float, float, float, float]
    action_magnitude: float
    adapter_query_count: int
    adapter_query_fresh: bool
    adapter_latency_ms: float | None
    adapter_error: str | None
    attached: bool
    eef_position: tuple[float, float, float]
    cube_position: tuple[float, float, float]
    cube_goal_position: tuple[float, float, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActionPlaygroundRecord:
    model_name: str
    task_id: str
    instruction: str
    gif_path: str
    runtime: str
    ok: bool
    frames: tuple[ActionPlaygroundFrame, ...]
    summary: dict[str, object]
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _tuple3(value: object, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return default


def _tuple4(
    value: object,
    default: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) >= 4:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    return default


def _float_tuple3(value: Sequence[float]) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _float_tuple4(value: Sequence[float]) -> tuple[float, float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _int_value(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return default


def _float_value(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _optional_float_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _bool_value(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _str_value(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _spec_from_manifest_result(result: object) -> PyBulletGifSpec | None:
    if not isinstance(result, dict):
        return None
    raw_spec = result.get("spec")
    if not isinstance(raw_spec, dict):
        return None
    model_name = raw_spec.get("model_name")
    task_id = raw_spec.get("task_id")
    instruction = raw_spec.get("instruction")
    out = raw_spec.get("out")
    if not isinstance(model_name, str) or not model_name:
        return None
    if not isinstance(task_id, str) or not task_id:
        return None
    if not isinstance(instruction, str) or not instruction:
        return None
    if not isinstance(out, str) or not out:
        return None
    runtime = raw_spec.get("runtime", "local")
    remote_url = raw_spec.get("remote_url", "http://localhost:8000")
    return PyBulletGifSpec(
        model_name=model_name,
        task_id=task_id,
        instruction=instruction,
        out=Path(out),
        runtime=runtime if isinstance(runtime, str) else "local",
        remote_url=remote_url if isinstance(remote_url, str) else "http://localhost:8000",
        model_call_every=_int_value(raw_spec.get("model_call_every"), 8),
        render_stride=_int_value(raw_spec.get("render_stride"), 8),
        cube_initial_position=_tuple3(raw_spec.get("cube_initial_position"), (0.58, -0.16, 0.035)),
        cube_goal_position=_tuple3(raw_spec.get("cube_goal_position"), (0.58, 0.22, 0.035)),
        goal_tolerance_m=_float_value(raw_spec.get("goal_tolerance_m"), 0.15),
    )


def load_playground_specs(manifest: Path) -> list[PyBulletGifSpec]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    raw_results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        return []
    specs: list[PyBulletGifSpec] = []
    for raw_result in raw_results:
        spec = _spec_from_manifest_result(raw_result)
        if spec is not None:
            specs.append(spec)
    return specs


def _displayed_action(sample: RenderSample) -> tuple[float, float, float, float]:
    return sample.adapter_action if sample.adapter_action is not None else sample.scripted_action


def _magnitude(action: tuple[float, float, float, float]) -> float:
    return float(sum(value * value for value in action) ** 0.5)


def samples_to_playground_frames(
    samples: Sequence[RenderSample],
) -> tuple[ActionPlaygroundFrame, ...]:
    frames: list[ActionPlaygroundFrame] = []
    for sample in samples:
        action = _displayed_action(sample)
        frames.append(
            ActionPlaygroundFrame(
                frame_index=sample.frame_index,
                phase=sample.phase,
                sim_time=sample.sim_time,
                scripted_action=_float_tuple4(sample.scripted_action),
                adapter_action=(
                    _float_tuple4(sample.adapter_action)
                    if sample.adapter_action is not None
                    else None
                ),
                displayed_action=_float_tuple4(action),
                action_magnitude=_magnitude(action),
                adapter_query_count=sample.adapter_query_count,
                adapter_query_fresh=sample.adapter_query_fresh,
                adapter_latency_ms=sample.adapter_latency_ms,
                adapter_error=sample.adapter_error,
                attached=sample.attached,
                eef_position=_float_tuple3(sample.position),
                cube_position=_float_tuple3(sample.cube_position),
                cube_goal_position=_float_tuple3(sample.cube_goal_position),
            )
        )
    return tuple(frames)


def _simulate_spec(spec: PyBulletGifSpec) -> list[RenderSample]:
    return run_simulation(spec.to_config())


def build_action_playground_records(
    manifest: Path,
    *,
    simulator: Simulator = _simulate_spec,
    max_records: int | None = None,
    allow_local_heavy: bool = False,
) -> list[ActionPlaygroundRecord]:
    specs = load_playground_specs(manifest)
    if max_records is not None:
        specs = specs[:max_records]

    records: list[ActionPlaygroundRecord] = []
    for spec in specs:
        if (
            not allow_local_heavy
            and spec.runtime == "local"
            and spec.model_name in HEAVY_LOCAL_MODELS
        ):
            records.append(
                ActionPlaygroundRecord(
                    model_name=spec.model_name,
                    task_id=spec.task_id,
                    instruction=spec.instruction,
                    gif_path=str(spec.out),
                    runtime=spec.runtime,
                    ok=False,
                    frames=(),
                    summary={},
                    error=(
                        "local heavy adapter skipped; use a remote manifest or "
                        "--allow-local-heavy"
                    ),
                )
            )
            continue
        try:
            samples = simulator(spec)
            summary = summarize_pybullet_samples(
                spec.model_name,
                spec.runtime,
                samples,
                task_id=spec.task_id,
                instruction=spec.instruction,
                goal_tolerance_m=spec.goal_tolerance_m,
                remote_url=spec.remote_url if spec.runtime == "remote" else None,
            )
            records.append(
                ActionPlaygroundRecord(
                    model_name=spec.model_name,
                    task_id=spec.task_id,
                    instruction=spec.instruction,
                    gif_path=str(spec.out),
                    runtime=spec.runtime,
                    ok=summary.ok,
                    frames=samples_to_playground_frames(samples),
                    summary=asdict(summary),
                    error=summary.last_error,
                )
            )
        except Exception as exc:
            records.append(
                ActionPlaygroundRecord(
                    model_name=spec.model_name,
                    task_id=spec.task_id,
                    instruction=spec.instruction,
                    gif_path=str(spec.out),
                    runtime=spec.runtime,
                    ok=False,
                    frames=(),
                    summary={},
                    error=str(exc),
                )
            )
    return records


def write_action_playground_trace(path: Path, records: Sequence[ActionPlaygroundRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": ACTION_PLAYGROUND_SCHEMA,
        "records": [record.to_dict() for record in records],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _frame_from_payload(payload: object) -> ActionPlaygroundFrame:
    if not isinstance(payload, Mapping):
        raise ValueError("action playground frame must be an object")
    adapter_action_raw = payload.get("adapter_action")
    return ActionPlaygroundFrame(
        frame_index=_int_value(payload.get("frame_index"), 0),
        phase=_str_value(payload.get("phase"), "unknown"),
        sim_time=_float_value(payload.get("sim_time"), 0.0),
        scripted_action=_tuple4(payload.get("scripted_action"), (0.0, 0.0, 0.0, 0.0)),
        adapter_action=(
            _tuple4(adapter_action_raw, (0.0, 0.0, 0.0, 0.0))
            if adapter_action_raw is not None
            else None
        ),
        displayed_action=_tuple4(payload.get("displayed_action"), (0.0, 0.0, 0.0, 0.0)),
        action_magnitude=_float_value(payload.get("action_magnitude"), 0.0),
        adapter_query_count=_int_value(payload.get("adapter_query_count"), 0),
        adapter_query_fresh=_bool_value(payload.get("adapter_query_fresh")),
        adapter_latency_ms=_optional_float_value(payload.get("adapter_latency_ms")),
        adapter_error=_str_value(payload.get("adapter_error")) or None,
        attached=_bool_value(payload.get("attached")),
        eef_position=_tuple3(payload.get("eef_position"), (0.0, 0.0, 0.0)),
        cube_position=_tuple3(payload.get("cube_position"), (0.0, 0.0, 0.0)),
        cube_goal_position=_tuple3(payload.get("cube_goal_position"), (0.0, 0.0, 0.0)),
    )


def _record_from_payload(payload: object) -> ActionPlaygroundRecord:
    if not isinstance(payload, Mapping):
        raise ValueError("action playground record must be an object")
    raw_frames = payload.get("frames", ())
    frames = (
        tuple(_frame_from_payload(frame) for frame in raw_frames)
        if isinstance(raw_frames, Sequence) and not isinstance(raw_frames, str)
        else ()
    )
    raw_summary = payload.get("summary", {})
    summary = dict(raw_summary) if isinstance(raw_summary, Mapping) else {}
    return ActionPlaygroundRecord(
        model_name=_str_value(payload.get("model_name"), "unknown"),
        task_id=_str_value(payload.get("task_id"), "unknown"),
        instruction=_str_value(payload.get("instruction")),
        gif_path=_str_value(payload.get("gif_path")),
        runtime=_str_value(payload.get("runtime"), "local"),
        ok=_bool_value(payload.get("ok")),
        frames=frames,
        summary=summary,
        error=_str_value(payload.get("error")) or None,
    )


def load_action_playground_trace(path: Path) -> list[ActionPlaygroundRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_records = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, str):
        raise ValueError(f"action playground trace has no records list: {path}")
    return [_record_from_payload(record) for record in raw_records]


def merge_action_playground_records(
    records: Sequence[ActionPlaygroundRecord],
) -> list[ActionPlaygroundRecord]:
    merged: list[ActionPlaygroundRecord] = []
    indexes: dict[tuple[str, str, str], int] = {}
    for record in records:
        key = (record.task_id, record.model_name, record.runtime)
        existing_index = indexes.get(key)
        if existing_index is None:
            indexes[key] = len(merged)
            merged.append(record)
        else:
            merged[existing_index] = record
    return merged


def load_action_playground_traces(paths: Sequence[Path]) -> list[ActionPlaygroundRecord]:
    records: list[ActionPlaygroundRecord] = []
    for path in paths:
        records.extend(load_action_playground_trace(path))
    return merge_action_playground_records(records)


def _display_path(path: str, *, relative_to: Path | None) -> str:
    raw = Path(path)
    if relative_to is None:
        return raw.as_posix()
    try:
        return raw.relative_to(relative_to).as_posix()
    except ValueError:
        return raw.as_posix()


def _records_payload(
    records: Sequence[ActionPlaygroundRecord],
    *,
    path_relative_to: Path | None,
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for record in records:
        item = record.to_dict()
        item["gif_path"] = _display_path(record.gif_path, relative_to=path_relative_to)
        payload.append(item)
    return payload


def _json_script_payload(payload: object) -> str:
    return (
        json.dumps(payload, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def format_action_playground_html(
    records: Sequence[ActionPlaygroundRecord],
    *,
    title: str = "vla_zoo Action Playground",
    path_relative_to: Path | None = None,
) -> str:
    payload = {
        "title": title,
        "records": _records_payload(records, path_relative_to=path_relative_to),
    }
    data_json = _json_script_payload(payload)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; }}
    body {{ margin: 0; background: #f8fafc; color: #111827; }}
    header {{ padding: 28px min(5vw, 64px); background: #111827; color: #f9fafb; }}
    main {{ padding: 24px min(5vw, 64px) 44px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 44px); }}
    .layout {{ display: grid; grid-template-columns: minmax(280px, 42vw) 1fr; gap: 18px; }}
    .panel {{ background: #fff; border: 1px solid #dbe3ef; border-radius: 8px; padding: 14px; }}
    .controls {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }}
    .comparison {{ margin-bottom: 18px; }}
    .section-title {{ display: flex; align-items: baseline; justify-content: space-between;
      gap: 12px; margin-bottom: 10px; }}
    .section-title h2 {{ margin: 0; font-size: 20px; }}
    .section-title span {{ color: #64748b; font-size: 13px; }}
    .compare-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px; margin-bottom: 12px; }}
    .compare-card {{ border: 1px solid #dbe3ef; border-radius: 8px; padding: 9px;
      background: #f8fafc; cursor: pointer; }}
    .compare-card.selected {{ border-color: #0891b2; box-shadow: inset 0 0 0 2px #0891b2; }}
    .compare-card img {{ border-radius: 6px; margin-bottom: 8px; }}
    .compare-head {{ display: flex; align-items: center; justify-content: space-between;
      gap: 8px; margin-bottom: 7px; }}
    .compare-head strong {{ font-size: 15px; }}
    .status {{ border-radius: 999px; padding: 3px 7px; font-size: 12px; font-weight: 700; }}
    .status.ok {{ background: #dcfce7; color: #14532d; }}
    .status.miss {{ background: #fee2e2; color: #991b1b; }}
    .compare-meta {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px; font-size: 12px; color: #475569; }}
    .compare-meta b {{ display: block; color: #111827; font-size: 13px; }}
    label {{ display: grid; gap: 4px; color: #475569; font-size: 13px; }}
    select, input[type="range"] {{ accent-color: #0891b2; }}
    select {{ min-width: 180px; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; }}
    img {{ width: 100%; aspect-ratio: 16 / 9; object-fit: cover; border-radius: 8px; }}
    canvas {{ width: 100%; height: 240px; border: 1px solid #dbe3ef; border-radius: 8px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px; margin-top: 12px; }}
    .metric {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 7px; padding: 10px; }}
    .metric span {{ display: block; color: #64748b; font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 18px; }}
    .bars {{ display: grid; gap: 8px; margin-top: 12px; }}
    .bar {{ display: grid; grid-template-columns: 52px 1fr 64px; align-items: center; gap: 8px; }}
    .track {{ height: 12px; background: #e2e8f0; border-radius: 6px;
      position: relative; overflow: hidden; }}
    .fill {{ height: 100%; position: absolute; left: 50%; background: #0891b2; }}
    .fill.neg {{ left: auto; right: 50%; background: #9333ea; }}
    .truth {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px; margin-top: 18px; }}
    .truth .panel {{ background: #fefefe; }}
    pre {{ overflow: auto; background: #0f172a; color: #e2e8f0; padding: 12px;
      border-radius: 8px; }}
    @media (max-width: 920px) {{ .layout {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <p>Scrub PyBullet runtime traces beside generated GIFs. This visualizes adapter
    action behavior; it is not a VLA skill benchmark or hardware safety proof.</p>
  </header>
  <main>
    <section class="controls panel">
      <label>Task<select id="task"></select></label>
      <label>Adapter<select id="model"></select></label>
      <label style="min-width:260px">Trace frame
        <input id="scrubber" type="range" min="0" max="0" value="0">
      </label>
      <div class="metric"><span>Selected</span><strong id="selected">-</strong></div>
    </section>
    <section class="panel comparison">
      <div class="section-title">
        <h2>Task Comparison</h2>
        <span id="task-summary">-</span>
      </div>
      <div id="comparison-grid" class="compare-grid"></div>
      <canvas id="comparison-chart" width="960" height="260"></canvas>
    </section>
    <section class="layout">
      <div class="panel">
        <img id="gif" alt="selected PyBullet simulation GIF">
        <div class="metrics">
          <div class="metric"><span>Phase</span><strong id="phase">-</strong></div>
          <div class="metric"><span>Magnitude</span><strong id="magnitude">-</strong></div>
          <div class="metric"><span>Queries</span><strong id="queries">-</strong></div>
          <div class="metric"><span>Latency</span><strong id="latency">-</strong></div>
        </div>
        <div id="bars" class="bars"></div>
      </div>
      <div class="panel">
        <canvas id="chart" width="960" height="320"></canvas>
        <pre id="frame-json">{{}}</pre>
      </div>
    </section>
    <section class="truth">
      <div class="panel">
        <h2>What This Shows</h2>
        <ul>
          <li>Phase transitions from the PyBullet scripted scene.</li>
          <li>Action vectors passed through the vla_zoo adapter boundary.</li>
          <li>Adapter query timing, query freshness, and action magnitude.</li>
        </ul>
      </div>
      <div class="panel">
        <h2>What This Does Not Show</h2>
        <ul>
          <li>Real robot policy quality.</li>
          <li>OpenVLA/pi0/SmolVLA task success unless those traces are explicitly recorded.</li>
          <li>Hardware-safe actuation or calibrated robot bridges.</li>
        </ul>
      </div>
    </section>
  </main>
  <script id="payload" type="application/json">{data_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const records = payload.records || [];
    const dims = ["dx", "dy", "dz", "grip"];
    const taskSelect = document.getElementById("task");
    const modelSelect = document.getElementById("model");
    const scrubber = document.getElementById("scrubber");
    const byTask = new Map();
    for (const record of records) {{
      if (!byTask.has(record.task_id)) byTask.set(record.task_id, []);
      byTask.get(record.task_id).push(record);
    }}
    function option(select, value, label) {{
      const item = document.createElement("option");
      item.value = value;
      item.textContent = label;
      select.appendChild(item);
    }}
    for (const task of byTask.keys()) option(taskSelect, task, task);
    function currentRecords() {{ return byTask.get(taskSelect.value) || []; }}
    function populateModels() {{
      modelSelect.innerHTML = "";
      for (const record of currentRecords()) {{
        option(modelSelect, record.model_name, record.model_name);
      }}
    }}
    function currentRecord() {{
      return currentRecords().find(
        (record) => record.model_name === modelSelect.value,
      ) || currentRecords()[0];
    }}
    function fmt(value, digits = 3) {{
      return Number.isFinite(value) ? Number(value).toFixed(digits) : "-";
    }}
    function metric(summary, key, digits = 3) {{
      const value = summary && summary[key];
      return Number.isFinite(value) ? fmt(value, digits) : "-";
    }}
    function drawComparisonChart(taskRecords, selectedName) {{
      const canvas = document.getElementById("comparison-chart");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const pad = 34;
      const colors = ["#0891b2", "#9333ea", "#16a34a", "#f97316", "#dc2626"];
      ctx.strokeStyle = "#cbd5e1";
      ctx.beginPath();
      ctx.moveTo(pad, canvas.height - pad);
      ctx.lineTo(canvas.width - pad, canvas.height - pad);
      ctx.moveTo(pad, pad);
      ctx.lineTo(pad, canvas.height - pad);
      ctx.stroke();
      const maxMag = Math.max(
        0.1,
        ...taskRecords.flatMap(
          (record) => (record.frames || []).map((frame) => frame.action_magnitude || 0),
        ),
      );
      taskRecords.forEach((record, recordIndex) => {{
        const frames = record.frames || [];
        ctx.strokeStyle = colors[recordIndex % colors.length];
        ctx.lineWidth = record.model_name === selectedName ? 4 : 2;
        ctx.beginPath();
        frames.forEach((frame, index) => {{
          const x = pad + index / Math.max(1, frames.length - 1) * (canvas.width - pad * 2);
          const y = canvas.height - pad - (frame.action_magnitude || 0) / maxMag
            * (canvas.height - pad * 2);
          if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        ctx.stroke();
        ctx.fillStyle = colors[recordIndex % colors.length];
        ctx.fillText(record.model_name, pad + recordIndex * 92, 18);
      }});
    }}
    function renderComparison(selectedRecord) {{
      const taskRecords = currentRecords();
      const grid = document.getElementById("comparison-grid");
      const okCount = taskRecords.filter((record) => record.ok).length;
      document.getElementById("task-summary").textContent =
        `${{okCount}}/${{taskRecords.length}} records ok`;
      grid.innerHTML = "";
      for (const record of taskRecords) {{
        const summary = record.summary || {{}};
        const card = document.createElement("button");
        card.type = "button";
        card.className = "compare-card";
        if (record.model_name === selectedRecord.model_name) {{
          card.classList.add("selected");
        }}
        card.addEventListener("click", () => {{
          modelSelect.value = record.model_name;
          scrubber.value = 0;
          render();
        }});
        const image = document.createElement("img");
        image.src = record.gif_path;
        image.alt = `${{record.model_name}} PyBullet GIF`;
        const head = document.createElement("div");
        head.className = "compare-head";
        const name = document.createElement("strong");
        name.textContent = record.model_name;
        const status = document.createElement("span");
        status.className = record.ok ? "status ok" : "status miss";
        status.textContent = record.ok ? "ok" : "check";
        head.append(name, status);
        const meta = document.createElement("div");
        meta.className = "compare-meta";
        const items = [
          ["task", summary.task_success ? "success" : "miss"],
          ["goal m", metric(summary, "final_cube_distance_to_goal")],
          ["queries", summary.adapter_queries ?? "-"],
          ["mean |a|", metric(summary, "mean_abs_action")],
          ["latency", `${{metric(summary, "mean_latency_ms", 2)}} ms`],
          ["errors", summary.adapter_errors ?? "-"],
        ];
        for (const [label, value] of items) {{
          const item = document.createElement("span");
          const bold = document.createElement("b");
          bold.textContent = value;
          item.textContent = label;
          item.prepend(bold);
          meta.appendChild(item);
        }}
        card.append(image, head, meta);
        grid.appendChild(card);
      }}
      drawComparisonChart(taskRecords, selectedRecord.model_name);
    }}
    function drawChart(record, selectedIndex) {{
      const canvas = document.getElementById("chart");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const frames = record.frames || [];
      const pad = 36;
      ctx.strokeStyle = "#cbd5e1";
      ctx.beginPath();
      ctx.moveTo(pad, canvas.height / 2);
      ctx.lineTo(canvas.width - pad, canvas.height / 2);
      ctx.stroke();
      const colors = ["#0891b2", "#9333ea", "#16a34a", "#f97316"];
      dims.forEach((dim, dimIndex) => {{
        ctx.strokeStyle = colors[dimIndex];
        ctx.lineWidth = 2;
        ctx.beginPath();
        frames.forEach((frame, index) => {{
          const action = frame.displayed_action || [0, 0, 0, 0];
          const x = pad + index / Math.max(1, frames.length - 1) * (canvas.width - pad * 2);
          const y = canvas.height / 2 - Math.max(-1, Math.min(1, action[dimIndex] || 0)) * 120;
          if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        ctx.stroke();
        ctx.fillStyle = colors[dimIndex];
        ctx.fillText(dim, pad + dimIndex * 58, 18);
      }});
      const x = pad + selectedIndex / Math.max(1, frames.length - 1) * (canvas.width - pad * 2);
      ctx.strokeStyle = "#ef4444";
      ctx.beginPath();
      ctx.moveTo(x, 28);
      ctx.lineTo(x, canvas.height - 18);
      ctx.stroke();
    }}
    function renderBars(action) {{
      const bars = document.getElementById("bars");
      bars.innerHTML = "";
      dims.forEach((dim, index) => {{
        const value = Math.max(-1, Math.min(1, Number(action[index] || 0)));
        const row = document.createElement("div");
        row.className = "bar";
        const label = document.createElement("span");
        label.textContent = dim;
        const track = document.createElement("div");
        track.className = "track";
        const fill = document.createElement("div");
        fill.className = value < 0 ? "fill neg" : "fill";
        fill.style.width = `${{Math.abs(value) * 50}}%`;
        track.appendChild(fill);
        const number = document.createElement("code");
        number.textContent = fmt(value);
        row.append(label, track, number);
        bars.appendChild(row);
      }});
    }}
    function render() {{
      const record = currentRecord();
      if (!record) return;
      const frames = record.frames || [];
      const index = Math.min(Number(scrubber.value), Math.max(0, frames.length - 1));
      const frame = frames[index] || {{}};
      scrubber.max = Math.max(0, frames.length - 1);
      renderComparison(record);
      document.getElementById("gif").src = record.gif_path;
      document.getElementById("selected").textContent =
        `${{record.task_id}} / ${{record.model_name}}`;
      document.getElementById("phase").textContent = frame.phase || "-";
      document.getElementById("magnitude").textContent = fmt(frame.action_magnitude);
      document.getElementById("queries").textContent = frame.adapter_query_count ?? "-";
      document.getElementById("latency").textContent =
        frame.adapter_latency_ms == null ? "-" : `${{fmt(frame.adapter_latency_ms, 1)}} ms`;
      renderBars(frame.displayed_action || [0, 0, 0, 0]);
      drawChart(record, index);
      document.getElementById("frame-json").textContent = JSON.stringify(frame, null, 2);
    }}
    taskSelect.addEventListener("change", () => {{
      populateModels();
      scrubber.value = 0;
      render();
    }});
    modelSelect.addEventListener("change", () => {{ scrubber.value = 0; render(); }});
    scrubber.addEventListener("input", render);
    populateModels();
    render();
  </script>
</body>
</html>
"""
    return "\n".join(line.rstrip() for line in html.splitlines()) + "\n"


def write_action_playground_html(
    path: Path,
    records: Sequence[ActionPlaygroundRecord],
    *,
    title: str = "vla_zoo Action Playground",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_action_playground_html(records, title=title, path_relative_to=path.parent),
        encoding="utf-8",
    )
