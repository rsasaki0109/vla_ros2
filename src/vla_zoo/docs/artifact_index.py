from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

# Curated catalog of the visible README/Pages artifacts. This is intentionally a
# hand-maintained list (not a directory crawl) so each entry can carry honest
# status and evidence caveats. Paths are repository-relative.
CATEGORIES = (
    "model evidence",
    "simulation",
    "runtime dashboard",
    "ROS2",
    "adapter docs",
    "GPU probes",
)

# Provenance of an artifact relative to the checked-in tree.
#   generated -> reproducible from a CLI command
#   checked   -> recorded evidence committed into the repo
#   manual    -> hand-authored documentation
KINDS = ("generated", "checked", "manual")


@dataclass(frozen=True)
class ArtifactEntry:
    """One curated, linkable repository artifact."""

    title: str
    path: str
    category: str
    status: str
    kind: str
    source_command: str | None = None
    caveat: str | None = None
    exists: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def with_existence(self, *, root: Path) -> ArtifactEntry:
        return ArtifactEntry(
            title=self.title,
            path=self.path,
            category=self.category,
            status=self.status,
            kind=self.kind,
            source_command=self.source_command,
            caveat=self.caveat,
            exists=(root / self.path).exists(),
        )


@dataclass(frozen=True)
class ArtifactIndex:
    """A curated artifact catalog with on-disk existence resolved."""

    entries: tuple[ArtifactEntry, ...]

    @property
    def missing(self) -> tuple[ArtifactEntry, ...]:
        return tuple(entry for entry in self.entries if entry.exists is False)

    @property
    def ok_overall(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, object]:
        return {
            "count": len(self.entries),
            "missing": len(self.missing),
            "ok_overall": self.ok_overall,
            "categories": list(CATEGORIES),
            "entries": [entry.to_dict() for entry in self.entries],
        }


DEFAULT_ARTIFACTS: tuple[ArtifactEntry, ...] = (
    ArtifactEntry(
        title="VLA model evidence matrix (HTML)",
        path="docs/assets/vla_model_evidence_matrix.html",
        category="model evidence",
        status="generated",
        kind="generated",
        source_command=(
            "vla-zoo compare evidence --html-out docs/assets/vla_model_evidence_matrix.html"
        ),
        caveat="Runtime-path evidence only; not a policy-quality or robot task-success claim.",
    ),
    ArtifactEntry(
        title="VLA model evidence matrix (Markdown)",
        path="docs/assets/vla_model_evidence_matrix.md",
        category="model evidence",
        status="generated",
        kind="generated",
        source_command=(
            "vla-zoo compare evidence --markdown-out docs/assets/vla_model_evidence_matrix.md"
        ),
        caveat="Runtime-path evidence only.",
    ),
    ArtifactEntry(
        title="VLA model evidence matrix (JSON)",
        path="docs/assets/vla_model_evidence_matrix.json",
        category="model evidence",
        status="generated",
        kind="generated",
        source_command="vla-zoo compare evidence --out docs/assets/vla_model_evidence_matrix.json",
        caveat="Machine-readable evidence matrix.",
    ),
    ArtifactEntry(
        title="OpenVLA local runtime evidence",
        path="docs/openvla_local_runtime.md",
        category="model evidence",
        status="verified",
        kind="checked",
        source_command=(
            "PYTHONPATH=src python3 scripts/measure_openvla_runtime.py "
            "--out docs/assets/openvla_local_runtime.json"
        ),
        caveat=(
            "Measured load/VRAM/latency for OpenVLA-7b (4-bit) on a synthetic frame; "
            "a runtime-path claim, not task success."
        ),
    ),
    ArtifactEntry(
        title="OpenVLA local runtime measurements (JSON)",
        path="docs/assets/openvla_local_runtime.json",
        category="model evidence",
        status="verified",
        kind="checked",
        source_command=(
            "PYTHONPATH=src python3 scripts/measure_openvla_runtime.py "
            "--out docs/assets/openvla_local_runtime.json"
        ),
        caveat="Machine-readable measured runtime profile (RTX 4070 Ti SUPER, 4-bit).",
    ),
    ArtifactEntry(
        title="SmolVLA local runtime evidence",
        path="docs/smolvla_local_runtime.md",
        category="model evidence",
        status="verified",
        kind="checked",
        source_command=(
            "PYTHONPATH=src python3 scripts/measure_lerobot_runtime.py "
            "--model smolvla --out docs/assets/smolvla_local_runtime.json"
        ),
        caveat=(
            "Measured load/VRAM/latency for lerobot/smolvla_base on a synthetic frame; "
            "a runtime-path claim, not task success."
        ),
    ),
    ArtifactEntry(
        title="SmolVLA local runtime measurements (JSON)",
        path="docs/assets/smolvla_local_runtime.json",
        category="model evidence",
        status="verified",
        kind="checked",
        source_command=(
            "PYTHONPATH=src python3 scripts/measure_lerobot_runtime.py "
            "--model smolvla --out docs/assets/smolvla_local_runtime.json"
        ),
        caveat="Machine-readable measured runtime profile (RTX 4070 Ti SUPER, ~0.97 GB).",
    ),
    ArtifactEntry(
        title="PyBullet GIF gallery",
        path="docs/assets/gif_suite/index.html",
        category="simulation",
        status="generated",
        kind="generated",
        source_command="vla-zoo demo gif-report --manifest docs/assets/gif_suite/gif_manifest.json",
        caveat="Deterministic runtime artifact; baselines, not robot skill.",
    ),
    ArtifactEntry(
        title="PyBullet baseline task verification",
        path="docs/assets/sample_task_verification/baseline_tasks.html",
        category="simulation",
        status="verified",
        kind="checked",
        source_command="vla-zoo compare tasks --models dummy,scripted,random --tasks all",
        caveat="Deterministic baselines, not VLA model-quality measurements.",
    ),
    ArtifactEntry(
        title="PyBullet comparison report",
        path="docs/assets/sample_compare_suite/pybullet_report.html",
        category="simulation",
        status="generated",
        kind="generated",
        source_command="vla-zoo compare suite --out-dir docs/assets/sample_compare_suite",
        caveat="Smoke-scene runtime telemetry.",
    ),
    ArtifactEntry(
        title="Benchmark result schema + ROS bag replay",
        path="docs/benchmark_results.md",
        category="simulation",
        status="manual",
        kind="manual",
        source_command=None,
        caveat="Versioned JSONL schema and replay stub; runtime-centric, no task-success claim.",
    ),
    ArtifactEntry(
        title="Runtime dashboard",
        path="docs/assets/sample_compare_suite/runtime_dashboard.html",
        category="runtime dashboard",
        status="generated",
        kind="generated",
        source_command=(
            "vla-zoo compare dashboard "
            "--results docs/assets/sample_compare_suite/pybullet_results.json"
        ),
        caveat="Static dashboard rendered from comparison results.",
    ),
    ArtifactEntry(
        title="ROS2 remote dummy dashboard",
        path="docs/assets/sample_ros2_remote_dummy/dashboard.html",
        category="runtime dashboard",
        status="generated",
        kind="generated",
        source_command="vla-zoo compare dashboard --status-log .../vla_status.jsonl",
        caveat="Rendered from recorded ROS2 dummy runtime logs.",
    ),
    ArtifactEntry(
        title="ROS2 remote smoke check",
        path="docs/assets/sample_ros2_remote_dummy/remote_smoke_check.md",
        category="ROS2",
        status="verified",
        kind="checked",
        source_command=(
            "vla-zoo ros remote-smoke-check --output-dir docs/assets/sample_ros2_remote_dummy"
        ),
        caveat="ROS2 remote dummy runtime evidence; launch stays dry-run safe by default.",
    ),
    ArtifactEntry(
        title="ROS2 remote smoke check (SmolVLA)",
        path="docs/assets/sample_ros2_remote_smolvla/remote_smoke_check.md",
        category="ROS2",
        status="verified",
        kind="checked",
        source_command=(
            "python3 scripts/record_ros2_remote_trace.py --model smolvla && vla-zoo ros "
            "remote-smoke-check --output-dir docs/assets/sample_ros2_remote_smolvla "
            "--model smolvla"
        ),
        caveat=(
            "Real SmolVLA server driven through the ROS2 runtime node (RemoteVLAClient); "
            "runtime-path evidence, not task success."
        ),
    ),
    ArtifactEntry(
        title="ROS2 remote smoke plan",
        path="docs/assets/ros2_remote_smoke_plan.md",
        category="ROS2",
        status="generated",
        kind="generated",
        source_command=(
            "vla-zoo ros remote-smoke-plan --markdown-out docs/assets/ros2_remote_smoke_plan.md"
        ),
        caveat="Command plan, not a recorded run.",
    ),
    ArtifactEntry(
        title="ROS2 action trace",
        path="docs/assets/sample_ros2_remote_dummy/action_trace.html",
        category="ROS2",
        status="generated",
        kind="generated",
        source_command="vla-zoo ros action-trace --action-log .../vla_actions.jsonl",
        caveat="Typed action messages only; core never commands motors.",
    ),
    ArtifactEntry(
        title="Jetson + remote GPU deployment guide",
        path="docs/deployment.md",
        category="ROS2",
        status="manual",
        kind="manual",
        source_command=None,
        caveat="Topology + serving + guards + bridges; runtime-centric, no task-success claim.",
    ),
    ArtifactEntry(
        title="ROS2 action replay summary",
        path="docs/assets/sample_benchmark/ros2_replay_summary.md",
        category="ROS2",
        status="generated",
        kind="generated",
        source_command=(
            "vla-zoo bench-replay --action-log .../vla_actions.jsonl "
            "--summary-md docs/assets/sample_benchmark/ros2_replay_summary.md"
        ),
        caveat="Latency/action-rate from a recorded action log; no task-success claim.",
    ),
    ArtifactEntry(
        title="Benchmark comparison report",
        path="docs/assets/sample_benchmark/benchmark_report.html",
        category="runtime dashboard",
        status="generated",
        kind="generated",
        source_command=(
            "vla-zoo bench-report --summaries .../ros2_replay_summary.json "
            "--html-out docs/assets/sample_benchmark/benchmark_report.html"
        ),
        caveat="Latency/action-rate comparison across summaries; no task-success claim.",
    ),
    ArtifactEntry(
        title="SmolVLA remote serving plan",
        path="docs/assets/smolvla_remote_smoke_plan.md",
        category="adapter docs",
        status="generated",
        kind="generated",
        source_command=(
            "vla-zoo smolvla-remote-plan "
            "--markdown-out docs/assets/smolvla_remote_smoke_plan.md"
        ),
        caveat="Isolated-env command plan, not a recorded /v1/predict run.",
    ),
    ArtifactEntry(
        title="SmolVLA remote environment isolation",
        path="docs/smolvla_remote.md",
        category="adapter docs",
        status="manual",
        kind="manual",
        source_command=None,
        caveat="Operations guidance; no policy-quality claim.",
    ),
    ArtifactEntry(
        title="OpenVLA remote GPU path",
        path="docs/openvla_remote.md",
        category="adapter docs",
        status="manual",
        kind="manual",
        source_command=None,
        caveat="Remote serving + health-first probe guidance; no task-success claim.",
    ),
    ArtifactEntry(
        title="pi0 remote-first path",
        path="docs/pi0_remote.md",
        category="adapter docs",
        status="manual",
        kind="manual",
        source_command=None,
        caveat="LeRobot/openpi version-compatibility and remote guidance; no task-success claim.",
    ),
    ArtifactEntry(
        title="pi0 server plan",
        path="docs/assets/pi0_server_plan.md",
        category="adapter docs",
        status="generated",
        kind="generated",
        source_command=(
            "vla-zoo serve-plan --models pi0 --markdown-out docs/assets/pi0_server_plan.md"
        ),
        caveat="Command plan, not a recorded /v1/predict run.",
    ),
    ArtifactEntry(
        title="GR00T blocked-status path",
        path="docs/groot_remote.md",
        category="adapter docs",
        status="manual",
        kind="manual",
        source_command=None,
        caveat=(
            "Blocked until the NVIDIA Isaac GR00T stack; no inference and no task-success claim."
        ),
    ),
    ArtifactEntry(
        title="Adapter cards index",
        path="docs/adapters/README.md",
        category="adapter docs",
        status="generated",
        kind="generated",
        source_command="vla-zoo compare cards --out-dir docs/adapters",
        caveat="Generated from registry metadata.",
    ),
    ArtifactEntry(
        title="Model comparison report",
        path="docs/reports/model_comparison.md",
        category="adapter docs",
        status="manual",
        kind="manual",
        source_command=None,
        caveat="Narrative comparison of runtime paths.",
    ),
    ArtifactEntry(
        title="SmolVLA GPU inference-path probe",
        path="docs/assets/sample_task_verification/smolvla_gpu_probe.md",
        category="GPU probes",
        status="verified",
        kind="checked",
        source_command=None,
        caveat=(
            "GPU inference-path probe with lerobot/smolvla_base; "
            "not a robot task-success claim."
        ),
    ),
    ArtifactEntry(
        title="OpenVLA prompt probe",
        path="docs/assets/sample_task_verification/openvla_prompt_probe.md",
        category="GPU probes",
        status="verified",
        kind="checked",
        source_command=None,
        caveat=(
            "Originally bf16-OOM'd on a 16 GB card; resolved via 4-bit. See the measured "
            "OpenVLA local runtime evidence."
        ),
    ),
    ArtifactEntry(
        title="Remote probe tool sample (dummy)",
        path="docs/assets/sample_task_verification/remote_probe_dummy.md",
        category="GPU probes",
        status="verified",
        kind="checked",
        source_command="vla-zoo remote-probe --model dummy --remote-url http://127.0.0.1:PORT",
        caveat="Verifies the health-first remote probe tool; not an OpenVLA claim.",
    ),
    ArtifactEntry(
        title="SmolVLA remote /v1/predict probe",
        path="docs/assets/sample_task_verification/smolvla_remote_probe.md",
        category="GPU probes",
        status="verified",
        kind="checked",
        source_command=(
            "vla-zoo serve --model smolvla --pretrained lerobot/smolvla_base --device cuda "
            "&& vla-zoo remote-probe --model smolvla --remote-url http://127.0.0.1:PORT"
        ),
        caveat=(
            "Real SmolVLA server returned a typed 6-DoF action over HTTP; runtime-path "
            "evidence, not task success."
        ),
    ),
    ArtifactEntry(
        title="OpenVLA remote /v1/predict probe",
        path="docs/assets/sample_task_verification/openvla_remote_probe.md",
        category="GPU probes",
        status="verified",
        kind="checked",
        source_command=(
            "vla-zoo serve --model openvla --pretrained openvla/openvla-7b --device cuda:0 "
            "--load-in-4bit && vla-zoo remote-probe --model openvla "
            "--remote-url http://127.0.0.1:PORT"
        ),
        caveat=(
            "Real OpenVLA-7b 4-bit server returned a typed 7-DoF action over HTTP; "
            "runtime-path evidence, not task success."
        ),
    ),
    ArtifactEntry(
        title="pi0 compatibility probe",
        path="docs/assets/sample_task_verification/pi0_compatibility_probe.md",
        category="GPU probes",
        status="planned",
        kind="checked",
        source_command=None,
        caveat="pi0/openpi is remote-first and still needs a recorded real action probe.",
    ),
)


def build_artifact_index(
    entries: Sequence[ArtifactEntry] = DEFAULT_ARTIFACTS,
    *,
    root: Path = Path("."),
) -> ArtifactIndex:
    """Resolve on-disk existence for each curated artifact entry."""

    resolved = tuple(entry.with_existence(root=root) for entry in entries)
    return ArtifactIndex(entries=resolved)


def artifact_index_payload(index: ArtifactIndex) -> dict[str, object]:
    """Return a machine-readable payload for the artifact index."""

    return index.to_dict()


def _entries_by_category(index: ArtifactIndex) -> list[tuple[str, list[ArtifactEntry]]]:
    grouped: list[tuple[str, list[ArtifactEntry]]] = []
    for category in CATEGORIES:
        members = [entry for entry in index.entries if entry.category == category]
        if members:
            grouped.append((category, members))
    return grouped


def _href_for(entry: ArtifactEntry, *, root: Path, html_dir: Path | None) -> str:
    """Resolve the artifact path to an href relative to the HTML output directory."""

    if html_dir is None:
        return entry.path
    target = root / entry.path
    return os.path.relpath(target, html_dir)


def format_artifact_index_html(
    index: ArtifactIndex,
    *,
    title: str = "vla_zoo Artifact Index",
    root: Path = Path("."),
    html_dir: Path | None = None,
) -> str:
    """Render a self-contained HTML artifact index grouped by category.

    When ``html_dir`` is given, hrefs are made relative to that directory so the
    page works when opened from where it is written; otherwise repo-relative
    paths are used.
    """

    rows: list[str] = []
    for category, members in _entries_by_category(index):
        rows.append(f'<tr class="category"><th colspan="5">{escape(category)}</th></tr>')
        for entry in members:
            missing = entry.exists is False
            href = _href_for(entry, root=root, html_dir=html_dir)
            link = (
                f'<a href="{escape(href)}">{escape(entry.title)}</a>'
                if not missing
                else f"{escape(entry.title)} <span class=\"missing\">(missing)</span>"
            )
            command = (
                f"<code>{escape(entry.source_command)}</code>" if entry.source_command else "-"
            )
            caveat = escape(entry.caveat) if entry.caveat else "-"
            rows.append(
                "<tr>"
                f"<td>{link}</td>"
                f'<td><span class="status">{escape(entry.status)}</span></td>'
                f"<td>{escape(entry.kind)}</td>"
                f"<td>{command}</td>"
                f"<td>{caveat}</td>"
                "</tr>"
            )
    summary = (
        f"{index.to_dict()['count']} artifacts, {len(index.missing)} missing"
        if index.missing
        else f"{index.to_dict()['count']} artifacts, all present"
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{escape(title)}</title>\n"
        "<style>\n"
        "body{font-family:system-ui,sans-serif;margin:2rem;color:#1f2933;}\n"
        "table{border-collapse:collapse;width:100%;}\n"
        "th,td{border:1px solid #d2d6dc;padding:0.4rem 0.6rem;text-align:left;"
        "vertical-align:top;font-size:0.9rem;}\n"
        "tr.category th{background:#1f2933;color:#fff;}\n"
        ".status{font-weight:600;}\n"
        ".missing{color:#b91c1c;font-weight:600;}\n"
        "code{font-size:0.8rem;word-break:break-all;}\n"
        "</style>\n</head>\n<body>\n"
        f"<h1>{escape(title)}</h1>\n"
        f"<p>{escape(summary)}. Statuses and caveats reflect runtime evidence only.</p>\n"
        '<table>\n<thead><tr><th>Artifact</th><th>Status</th><th>Kind</th>'
        "<th>Source command</th><th>Caveat</th></tr></thead>\n<tbody>\n"
        + "\n".join(rows)
        + "\n</tbody>\n</table>\n</body>\n</html>\n"
    )


def format_artifact_index_table(index: ArtifactIndex) -> str:
    """Render a human-readable status table for the artifact index."""

    lines = [
        f"{'category':<18} {'status':<10} {'kind':<10} {'present':<8} title",
        f"{'-' * 18} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 40}",
    ]
    for entry in index.entries:
        present = "yes" if entry.exists else "NO"
        lines.append(
            f"{entry.category:<18} {entry.status:<10} {entry.kind:<10} {present:<8} {entry.title}"
        )
    lines.append("")
    lines.append(f"count={len(index.entries)} missing={len(index.missing)}")
    return "\n".join(lines)
