"""A shareable, honest VLA *runtime* leaderboard.

This joins ranked benchmark summaries (latency / action throughput, produced by
``rank_summaries``) with a small curated table of recorded runtime metadata —
measured memory footprint, runtime-evidence status, and a link to the recorded
probe — into one scannable page (Markdown / JSON / standalone HTML).

It ranks adapters by a *runtime* metric, never by robot task-success or policy
quality. Adapters with no runtime measurement (because their local path is
currently blocked) are listed honestly as unranked ``blocked`` rows rather than
omitted, so the leaderboard is complete instead of cherry-picked. Every number is
sourced from a recorded artifact; nothing here is synthesised.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

from vla_zoo.benchmark.aggregate import RANKABLE_METRICS, rank_summaries
from vla_zoo.benchmark.results import BenchmarkSummary

#: Schema identifier for the leaderboard artifact.
LEADERBOARD_SCHEMA_VERSION = "vla-zoo-leaderboard/v1"

DEFAULT_METRIC = "latency_ms_p50"

_DISCLAIMER = (
    "Runtime leaderboard. Rank is by the selected latency / action-rate metric, measured "
    "on recorded probes — NOT by robot task-success or policy quality (policy_quality stays "
    "not_verified for every model). Memory is a measured runtime footprint. Blocked rows are "
    "adapters whose local runtime path is currently gated, shown for completeness rather than "
    "omitted; they carry no fabricated numbers."
)


@dataclass(frozen=True)
class RuntimeProfile:
    """Curated, *recorded* runtime metadata for one adapter (not auto-derived).

    ``memory_gb`` is a measured peak footprint; ``evidence_link`` points at the
    recorded probe that backs the row (relative to the leaderboard artifact's
    directory). ``status`` mirrors the runtime-evidence vocabulary used by the VLA
    evidence matrix (``verified`` / ``partial`` / ``blocked`` / ``planned``).
    """

    status: str
    memory_gb: float | None
    memory_note: str
    evidence_link: str | None
    note: str


#: Recorded runtime metadata, keyed by adapter. Every figure is sourced from a
#: checked-in artifact (the real-scene action probes and the measured 16 GB-fit
#: table in ``docs/deployment.md``); links are relative to the leaderboard's own
#: directory (``docs/assets/leaderboard/``). Blocked adapters carry no latency.
RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    "smolvla": RuntimeProfile(
        status="verified",
        memory_gb=0.97,
        memory_note="local GPU, full precision",
        evidence_link="../sample_pybullet_smolvla/runtime_action_probe.md",
        note=(
            "Real-scene PyBullet action probe (6-DoF on rendered frames); "
            "bf16 --dtype serve recorded."
        ),
    ),
    "openvla": RuntimeProfile(
        status="verified",
        memory_gb=4.6,
        memory_note="4-bit (bitsandbytes nf4); bf16 weights ~15 GB OOM a 16 GB card",
        evidence_link="../sample_pybullet_openvla/runtime_action_probe.md",
        note=(
            "Real-scene PyBullet action probe (7-DoF on rendered frames); "
            "local 4-bit + remote + ROS2 traces recorded."
        ),
    ),
    "pi0": RuntimeProfile(
        status="blocked",
        memory_gb=8.9,
        memory_note="lerobot/pi0_base bf16 (measured during load); float32 config OOMs",
        evidence_link="../sample_task_verification/pi0_compatibility_probe.md",
        note=(
            "Version-matched checkpoint resolves; local inference blocked on the gated "
            "google/paligemma-3b-pt-224 tokenizer. Use the remote runtime."
        ),
    ),
    "groot": RuntimeProfile(
        status="blocked",
        memory_gb=None,
        memory_note="n/a (no local runtime)",
        evidence_link="../sample_task_verification/groot_block_probe.md",
        note=(
            "Adapter refuses rather than fabricating; no GR00T package on PyPI. "
            "Real runtime is the NVIDIA Isaac-GR00T GitHub stack."
        ),
    ),
}


@dataclass(frozen=True)
class LeaderboardEntry:
    """One adapter's row: its rank, runtime metrics, and recorded metadata."""

    rank: int | None
    model: str
    status: str
    metric: str
    metric_value: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    action_rate_hz: float | None
    sample_count: int
    memory_gb: float | None
    memory_note: str
    evidence_link: str | None
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "model": self.model,
            "status": self.status,
            "metric": self.metric,
            "metric_value": self.metric_value,
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_p95": self.latency_ms_p95,
            "action_rate_hz": self.action_rate_hz,
            "sample_count": self.sample_count,
            "memory_gb": self.memory_gb,
            "memory_note": self.memory_note,
            "evidence_link": self.evidence_link,
            "note": self.note,
        }


@dataclass(frozen=True)
class Leaderboard:
    """A ranked runtime leaderboard over several adapters."""

    metric: str
    lower_is_better: bool
    entries: tuple[LeaderboardEntry, ...]
    schema_version: str = LEADERBOARD_SCHEMA_VERSION

    @property
    def count(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "lower_is_better": self.lower_is_better,
            "count": self.count,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def build_leaderboard(
    summaries: Sequence[BenchmarkSummary],
    *,
    metric: str = DEFAULT_METRIC,
    profiles: dict[str, RuntimeProfile] | None = None,
    include_blocked: bool = True,
) -> Leaderboard:
    """Rank measured ``summaries`` and join each with its recorded runtime profile.

    Ranked rows come from :func:`rank_summaries`. When ``include_blocked`` is set, any
    adapter in ``profiles`` that has *no* measured summary is appended as an unranked
    row (``rank=None``) carrying its status and memory but no latency — so a blocked
    adapter is shown honestly rather than dropped. Raises on an unsupported metric.
    """

    table = RUNTIME_PROFILES if profiles is None else profiles
    report = rank_summaries(summaries, metric=metric)

    entries: list[LeaderboardEntry] = []
    measured: set[str] = set()
    for ranked in report.ranked:
        summary = ranked.summary
        measured.add(summary.model)
        profile = table.get(summary.model)
        entries.append(
            LeaderboardEntry(
                rank=ranked.rank,
                model=summary.model,
                status=profile.status if profile else "measured",
                metric=metric,
                metric_value=ranked.metric_value,
                latency_ms_p50=summary.latency_ms_p50,
                latency_ms_p95=summary.latency_ms_p95,
                action_rate_hz=summary.action_rate_hz,
                sample_count=summary.sample_count,
                memory_gb=profile.memory_gb if profile else None,
                memory_note=profile.memory_note if profile else "",
                evidence_link=profile.evidence_link if profile else None,
                note=profile.note if profile else "",
            )
        )

    if include_blocked:
        for model, profile in table.items():
            if model in measured:
                continue
            entries.append(
                LeaderboardEntry(
                    rank=None,
                    model=model,
                    status=profile.status,
                    metric=metric,
                    metric_value=None,
                    latency_ms_p50=None,
                    latency_ms_p95=None,
                    action_rate_hz=None,
                    sample_count=0,
                    memory_gb=profile.memory_gb,
                    memory_note=profile.memory_note,
                    evidence_link=profile.evidence_link,
                    note=profile.note,
                )
            )

    return Leaderboard(
        metric=metric,
        lower_is_better=RANKABLE_METRICS[metric],
        entries=tuple(entries),
    )


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _hz(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _gb(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f} GB"


_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}


def _rank_label(rank: int | None) -> str:
    if rank is None:
        return "—"
    return f"{_MEDALS.get(rank, '')} {rank}".strip()


def format_leaderboard_markdown(
    board: Leaderboard,
    *,
    title: str = "VLA Runtime Leaderboard",
) -> str:
    """Render the leaderboard as a Markdown table."""

    direction = "lower is better" if board.lower_is_better else "higher is better"
    headers = [
        "Rank",
        "Model",
        "Status",
        "Latency p50 (ms)",
        "Latency p95 (ms)",
        "Action rate (Hz)",
        "Memory",
        "Evidence",
    ]
    lines = [
        f"# {title}",
        "",
        f"- Schema: `{board.schema_version}`",
        f"- Ranked by: `{board.metric}` ({direction})",
        f"- Adapters: {board.count}",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for entry in board.entries:
        link = (
            f"[probe]({entry.evidence_link})" if entry.evidence_link else "-"
        )
        cells = [
            _rank_label(entry.rank),
            entry.model,
            entry.status,
            _ms(entry.latency_ms_p50),
            _ms(entry.latency_ms_p95),
            _hz(entry.action_rate_hz),
            _gb(entry.memory_gb),
            link,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", _DISCLAIMER, ""])
    return "\n".join(lines) + "\n"


_STATUS_CLASS = {
    "verified": "ok",
    "partial": "warn",
    "blocked": "bad",
    "planned": "muted",
    "measured": "ok",
}

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #0b1220; --panel: #131c2e; --text: #e8eefb; --muted: #93a3bd;
      --line: #233149; --accent: #4cc2ff; --head: #1a2740;
      --ok: #2fbf71; --warn: #e0a33e; --bad: #e2554f; --mute: #6b7a93;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    main { max-width: 1080px; margin: 0 auto; padding: 40px 20px 56px; }
    h1 { margin: 0 0 6px; font-size: clamp(28px, 5vw, 46px); letter-spacing: -0.5px; }
    .sub { color: var(--muted); line-height: 1.6; margin: 0 0 4px; }
    .sub code { background: #0f1830; padding: 1px 7px; border-radius: 6px; color: var(--accent); }
    .panel {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 16px; padding: 6px 8px; margin-top: 22px; overflow-x: auto;
    }
    table { width: 100%; border-collapse: collapse; font-size: 14.5px; }
    th, td {
      text-align: left; padding: 13px 14px;
      border-bottom: 1px solid var(--line); white-space: nowrap;
    }
    th { color: var(--accent); font-weight: 600; background: var(--head); }
    tbody tr:last-child td { border-bottom: none; }
    td.rank { font-size: 17px; font-weight: 700; }
    td.model { font-weight: 700; }
    a { color: var(--accent); }
    .badge {
      display: inline-block; padding: 2px 10px; border-radius: 999px;
      font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px;
    }
    .badge.ok { background: rgba(47,191,113,0.16); color: var(--ok); }
    .badge.warn { background: rgba(224,163,62,0.16); color: var(--warn); }
    .badge.bad { background: rgba(226,85,79,0.16); color: var(--bad); }
    .badge.muted { background: rgba(107,122,147,0.16); color: var(--mute); }
    .note { color: var(--muted); font-size: 13px; line-height: 1.6; margin-top: 20px; }
  </style>
</head>
<body>
  <main>
    <h1>__TITLE__ 🤖</h1>
    <p class="sub">Schema <code>__SCHEMA__</code> · ranked by
      <code>__METRIC__</code> (__DIRECTION__)</p>
    <p class="sub">Runtime path only — latency, throughput, and memory.
      <strong>Not</strong> a robot task-success or policy-quality ranking.</p>
    <div class="panel">
      <table>
        <thead><tr>__HEAD__</tr></thead>
        <tbody>
__ROWS__
        </tbody>
      </table>
    </div>
    <p class="note">__DISCLAIMER__</p>
  </main>
</body>
</html>
"""

_HTML_HEADERS = (
    "Rank",
    "Model",
    "Status",
    "Latency p50 (ms)",
    "Latency p95 (ms)",
    "Action rate (Hz)",
    "Memory",
    "Evidence",
)


def format_leaderboard_html(
    board: Leaderboard,
    *,
    title: str = "VLA Runtime Leaderboard",
) -> str:
    """Render the leaderboard as a polished, shareable standalone HTML page."""

    direction = "lower is better" if board.lower_is_better else "higher is better"
    head = "".join(f"<th>{escape(label)}</th>" for label in _HTML_HEADERS)
    rows = []
    for entry in board.entries:
        badge_class = _STATUS_CLASS.get(entry.status, "muted")
        status_html = f'<span class="badge {badge_class}">{escape(entry.status)}</span>'
        if entry.evidence_link:
            link_html = f'<a href="{escape(entry.evidence_link)}">probe</a>'
        else:
            link_html = "-"
        mem = _gb(entry.memory_gb)
        if entry.memory_gb is not None and entry.memory_note:
            mem = f'<span title="{escape(entry.memory_note)}">{escape(mem)}</span>'
        else:
            mem = escape(mem)
        cells = (
            f'<td class="rank">{escape(_rank_label(entry.rank))}</td>'
            f'<td class="model">{escape(entry.model)}</td>'
            f"<td>{status_html}</td>"
            f"<td>{escape(_ms(entry.latency_ms_p50))}</td>"
            f"<td>{escape(_ms(entry.latency_ms_p95))}</td>"
            f"<td>{escape(_hz(entry.action_rate_hz))}</td>"
            f"<td>{mem}</td>"
            f"<td>{link_html}</td>"
        )
        rows.append(f"          <tr>{cells}</tr>")

    return (
        _HTML_TEMPLATE.replace("__TITLE__", escape(title))
        .replace("__SCHEMA__", escape(board.schema_version))
        .replace("__METRIC__", escape(board.metric))
        .replace("__DIRECTION__", escape(direction))
        .replace("__HEAD__", head)
        .replace("__ROWS__", "\n".join(rows))
        .replace("__DISCLAIMER__", escape(_DISCLAIMER))
    )
