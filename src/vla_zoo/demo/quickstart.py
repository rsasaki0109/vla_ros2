"""A zero-dependency ``vla-zoo quickstart``: prove the runtime boundary in seconds.

A first-time user should be able to ``pip install vla_zoo`` and run one command that
exercises the real adapter/runtime boundary — ``load_model() -> predict() -> typed
action`` — on their own machine, with **no GPU, no model weights, and no PyBullet**.
It runs the pure-Python baseline adapters (``dummy`` / ``scripted`` / ``random``)
through the smoke benchmark, records per-adapter latency and a real typed action, and
renders a small local report that then points at the recorded real-VLA evidence.

This proves the *plumbing* works locally; it is explicitly not a model-quality claim.
The baselines are infrastructure baselines, not VLA policies.
"""

from __future__ import annotations

from dataclasses import dataclass

from vla_zoo.core.types import VLAAction, VLAActionChunk

#: Schema identifier for the quickstart report artifact.
QUICKSTART_SCHEMA_VERSION = "vla-zoo-quickstart/v1"

#: Pure-Python baselines that load and predict with only the core dependencies.
DEFAULT_QUICKSTART_MODELS = ("dummy", "scripted", "random")

#: Where to go once the boundary is proven locally — the recorded real evidence.
#: Absolute Pages URLs so the links work from any clone or copied report.
NEXT_STEPS: tuple[tuple[str, str], ...] = (
    (
        "VLA runtime leaderboard",
        "https://rsasaki0109.github.io/vla_zoo/assets/leaderboard/vla_runtime_leaderboard.html",
    ),
    (
        "VLA model evidence matrix",
        "https://rsasaki0109.github.io/vla_zoo/assets/vla_model_evidence_matrix.html",
    ),
    (
        "PyBullet GIF gallery",
        "https://rsasaki0109.github.io/vla_zoo/assets/gif_suite/",
    ),
    (
        "Full docs & demo site",
        "https://rsasaki0109.github.io/vla_zoo/",
    ),
)

_NOTE = (
    "Runtime-boundary smoke check on pure-Python baselines (dummy/scripted/random) — no GPU, "
    "weights, or PyBullet. It proves load_model() -> predict() -> typed action works locally and "
    "measures latency. Baselines are infrastructure baselines, NOT VLA policies; this is not a "
    "model-quality or task-success claim. See the linked evidence for real-adapter runtime paths."
)


@dataclass(frozen=True)
class QuickstartRow:
    """One baseline's locally-measured runtime-boundary result."""

    model: str
    action_space: str
    action_dim: int
    episodes: int
    latency_ms_p50: float | None
    latency_ms_mean: float | None
    action_rate_hz: float | None
    sample_action: tuple[float, ...]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "episodes": self.episodes,
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_mean": self.latency_ms_mean,
            "action_rate_hz": self.action_rate_hz,
            "sample_action": list(self.sample_action),
            "error": self.error,
        }


@dataclass(frozen=True)
class QuickstartReport:
    """A local runtime-boundary smoke report across several baselines."""

    rows: tuple[QuickstartRow, ...]
    episodes: int
    note: str = _NOTE
    schema_version: str = QUICKSTART_SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return bool(self.rows) and all(row.ok for row in self.rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "episodes": self.episodes,
            "ok": self.ok,
            "rows": [row.to_dict() for row in self.rows],
            "note": self.note,
        }


def _resolved_action(prediction: VLAAction | VLAActionChunk) -> VLAAction:
    return prediction.actions[0] if isinstance(prediction, VLAActionChunk) else prediction


def run_quickstart(
    models: tuple[str, ...] = DEFAULT_QUICKSTART_MODELS,
    *,
    episodes: int = 5,
) -> QuickstartReport:
    """Run each baseline through the smoke benchmark and capture a typed action.

    Uses only the core dependencies (the baselines are pure NumPy). A model that fails
    to load or predict becomes an error row rather than aborting the run, so one broken
    adapter never sinks the whole quickstart.
    """

    from vla_zoo.benchmark.results import summarize_records
    from vla_zoo.benchmark.runner import SmokeBenchmarkEnv, run_smoke_episode_records
    from vla_zoo.core.registry import load_model

    rows: list[QuickstartRow] = []
    for name in models:
        try:
            model = load_model(name)
            records, action_rate = run_smoke_episode_records(
                model, model_name=name, episodes=episodes
            )
            summary = summarize_records(records, action_rate_hz=action_rate)
            prediction = model.predict(observation=SmokeBenchmarkEnv().reset())
            action = _resolved_action(prediction)
            flat = [round(float(v), 6) for v in action.to_numpy().reshape(-1).tolist()]
            rows.append(
                QuickstartRow(
                    model=name,
                    action_space=action.spec.action_space,
                    action_dim=len(flat),
                    episodes=episodes,
                    latency_ms_p50=summary.latency_ms_p50,
                    latency_ms_mean=summary.latency_ms_mean,
                    action_rate_hz=summary.action_rate_hz,
                    sample_action=tuple(flat),
                )
            )
        except Exception as exc:  # noqa: BLE001 - recorded as an error row, not raised
            rows.append(
                QuickstartRow(
                    model=name,
                    action_space="-",
                    action_dim=0,
                    episodes=episodes,
                    latency_ms_p50=None,
                    latency_ms_mean=None,
                    action_rate_hz=None,
                    sample_action=(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return QuickstartReport(rows=tuple(rows), episodes=episodes)


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _hz(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _sample(values: tuple[float, ...]) -> str:
    if not values:
        return "-"
    shown = ", ".join(f"{v:.3f}" for v in values[:6])
    return f"[{shown}{', …' if len(values) > 6 else ''}]"


def format_quickstart_markdown(
    report: QuickstartReport,
    *,
    title: str = "vla_zoo quickstart",
) -> str:
    """Render the quickstart report as Markdown."""

    status = "✅ runtime boundary works" if report.ok else "⚠️ some baselines failed"
    headers = [
        "Model",
        "Action space",
        "Dim",
        "Latency p50 (ms)",
        "Latency mean (ms)",
        "Rate (Hz)",
        "Sample action",
    ]
    lines = [
        f"# {title}",
        "",
        f"- Schema: `{report.schema_version}`",
        f"- Episodes per adapter: {report.episodes}",
        f"- Status: {status}",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in report.rows:
        cells = [
            row.model,
            row.action_space,
            str(row.action_dim),
            _ms(row.latency_ms_p50),
            _ms(row.latency_ms_mean),
            _hz(row.action_rate_hz),
            "error" if row.error else _sample(row.sample_action),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Next steps", ""])
    for label, url in NEXT_STEPS:
        lines.append(f"- [{label}]({url})")
    lines.extend(["", report.note, ""])
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
      --bg: #f4f7fb; --panel: #fff; --text: #172033; --muted: #5b6b85;
      --line: #d8e2ee; --accent: #0f6f9f; --head: #eef4fb; --ok: #1f9d57;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    main { max-width: 940px; margin: 0 auto; padding: 38px 20px 52px; }
    h1 { margin: 0 0 4px; font-size: clamp(26px, 4vw, 40px); }
    h2 { margin: 26px 0 6px; font-size: 18px; color: var(--accent); }
    .status { font-size: 19px; font-weight: 700; color: var(--ok); margin: 8px 0 0; }
    p { color: var(--muted); line-height: 1.6; }
    .panel {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 14px; padding: 6px 10px; margin-top: 16px; overflow-x: auto;
    }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td {
      text-align: left; padding: 11px 12px;
      border-bottom: 1px solid var(--line); white-space: nowrap;
    }
    th { background: var(--head); color: var(--accent); font-weight: 600; }
    td:first-child { font-weight: 700; }
    tbody tr:last-child td { border-bottom: none; }
    .steps a {
      display: inline-block; margin: 4px 8px 4px 0; padding: 8px 14px;
      background: var(--accent); color: #fff; border-radius: 8px;
      text-decoration: none; font-weight: 600; font-size: 14px;
    }
    .note { font-size: 13px; margin-top: 18px; }
    code { background: #eef2f7; padding: 1px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <main>
    <h1>__TITLE__</h1>
    <p class="status">__STATUS__</p>
    <p>Schema <code>__SCHEMA__</code> · __EPISODES__ episodes per adapter ·
      pure-Python baselines, no GPU / weights / PyBullet.</p>
    <div class="panel">
      <table>
        <thead><tr>__HEAD__</tr></thead>
        <tbody>
__ROWS__
        </tbody>
      </table>
    </div>
    <h2>Next: real-adapter runtime evidence</h2>
    <div class="steps">__STEPS__</div>
    <p class="note">__NOTE__</p>
  </main>
</body>
</html>
"""

_HTML_HEADERS = (
    "Model",
    "Action space",
    "Dim",
    "Latency p50 (ms)",
    "Latency mean (ms)",
    "Rate (Hz)",
    "Sample action",
)


def format_quickstart_html(
    report: QuickstartReport,
    *,
    title: str = "vla_zoo quickstart",
) -> str:
    """Render the quickstart report as a friendly standalone HTML page."""

    from html import escape

    status = "✅ runtime boundary works on your machine" if report.ok else (
        "⚠️ some baselines failed — see the table"
    )
    head = "".join(f"<th>{escape(label)}</th>" for label in _HTML_HEADERS)
    rows = []
    for row in report.rows:
        sample = "error" if row.error else _sample(row.sample_action)
        cells = (
            f"<td>{escape(row.model)}</td>"
            f"<td>{escape(row.action_space)}</td>"
            f"<td>{row.action_dim}</td>"
            f"<td>{escape(_ms(row.latency_ms_p50))}</td>"
            f"<td>{escape(_ms(row.latency_ms_mean))}</td>"
            f"<td>{escape(_hz(row.action_rate_hz))}</td>"
            f"<td>{escape(sample)}</td>"
        )
        rows.append(f"          <tr>{cells}</tr>")
    steps = "".join(
        f'<a href="{escape(url)}">{escape(label)}</a>' for label, url in NEXT_STEPS
    )
    return (
        _HTML_TEMPLATE.replace("__TITLE__", escape(title))
        .replace("__STATUS__", escape(status))
        .replace("__SCHEMA__", escape(report.schema_version))
        .replace("__EPISODES__", str(report.episodes))
        .replace("__HEAD__", head)
        .replace("__ROWS__", "\n".join(rows))
        .replace("__STEPS__", steps)
        .replace("__NOTE__", escape(report.note))
    )
