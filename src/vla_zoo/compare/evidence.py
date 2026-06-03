from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from html import escape
from typing import Literal

from vla_zoo.core.registry import AdapterInfo, get_adapter_info

EvidenceStatus = Literal[
    "verified",
    "partial",
    "planned",
    "blocked",
    "not_verified",
    "not_applicable",
]

EVIDENCE_COLUMNS = (
    "contract",
    "local_runtime",
    "gpu_inference",
    "remote_server",
    "ros2_remote",
    "pybullet_tasks",
    "policy_quality",
)

COLUMN_LABELS = {
    "contract": "Contract",
    "local_runtime": "Local runtime",
    "gpu_inference": "GPU inference",
    "remote_server": "Remote server",
    "ros2_remote": "ROS2 remote",
    "pybullet_tasks": "PyBullet tasks",
    "policy_quality": "Policy quality",
}


@dataclass(frozen=True)
class EvidenceLink:
    label: str
    href: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceCell:
    status: EvidenceStatus
    summary: str
    links: tuple[EvidenceLink, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["links"] = [link.to_dict() for link in self.links]
        return payload


@dataclass(frozen=True)
class ModelEvidence:
    model: str
    upstream_project: str
    family: str
    adapter_status: str
    experimental: bool
    evidence: dict[str, EvidenceCell]
    next_step: str
    caveat: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "upstream_project": self.upstream_project,
            "family": self.family,
            "adapter_status": self.adapter_status,
            "experimental": self.experimental,
            "evidence": {key: value.to_dict() for key, value in self.evidence.items()},
            "next_step": self.next_step,
            "caveat": self.caveat,
        }


def _link(label: str, href: str) -> EvidenceLink:
    return EvidenceLink(label=label, href=href)


def _cell(
    status: EvidenceStatus,
    summary: str,
    links: Sequence[EvidenceLink] = (),
) -> EvidenceCell:
    return EvidenceCell(status=status, summary=summary, links=tuple(links))


def _registry_contract(info: AdapterInfo) -> EvidenceCell:
    return _cell(
        "verified",
        "Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats.",
        (_link("adapter card", f"../adapters/{info.name}.md"),),
    )


def _baseline_evidence(info: AdapterInfo) -> dict[str, EvidenceCell]:
    if info.name == "dummy":
        return {
            "contract": _registry_contract(info),
            "local_runtime": _cell(
                "verified",
                "CPU local predict path is covered by tests, CLI predict, and smoke reports.",
                (_link("action playground check", "../reports/model_comparison.md"),),
            ),
            "gpu_inference": _cell(
                "not_applicable",
                "Dummy is a neutral runtime baseline and does not require GPU inference.",
            ),
            "remote_server": _cell(
                "verified",
                "Temporary HTTP server returned typed actions through /v1/predict.",
                (_link("remote runtime smoke", "../reports/remote_runtime_smoke.md"),),
            ),
            "ros2_remote": _cell(
                "verified",
                "Recorded ROS2 node to RemoteVLAClient to HTTP dummy server path.",
                (_link("ROS2 remote evidence", "sample_ros2_remote_dummy/remote_smoke_check.md"),),
            ),
            "pybullet_tasks": _cell(
                "verified",
                "Recorded on all three deterministic PyBullet smoke tasks.",
                (_link("baseline task report", "sample_task_verification/baseline_tasks.md"),),
            ),
            "policy_quality": _cell(
                "not_applicable",
                "Dummy intentionally emits neutral actions; it is not a VLA policy.",
            ),
        }
    return {
        "contract": _registry_contract(info),
        "local_runtime": _cell(
            "verified",
            "CPU local predict path is exercised as a lightweight baseline.",
            (_link("action playground check", "../reports/model_comparison.md"),),
        ),
        "gpu_inference": _cell(
            "not_applicable",
            f"{info.name} is a lightweight baseline and does not require GPU inference.",
        ),
        "remote_server": _cell(
            "partial",
            "The runtime can serve this adapter, but the recorded remote smoke sample is "
            "dummy-only.",
            (_link("server plan", "sample_compare_suite/gpu_server_plan.md"),),
        ),
        "ros2_remote": _cell(
            "planned",
            "ROS2 remote recording path exists; no dedicated recorded sample is checked in yet.",
            (_link("ROS2 remote plan", "ros2_remote_smoke_plan.md"),),
        ),
        "pybullet_tasks": _cell(
            "verified",
            "Recorded on all three deterministic PyBullet smoke tasks.",
            (_link("baseline task report", "sample_task_verification/baseline_tasks.md"),),
        ),
        "policy_quality": _cell(
            "not_applicable",
            f"{info.name} is a baseline for runtime comparison, not a VLA model.",
        ),
    }


def _openvla_evidence(info: AdapterInfo) -> dict[str, EvidenceCell]:
    return {
        "contract": _registry_contract(info),
        "local_runtime": _cell(
            "verified",
            "OpenVLA-7b loaded and predicted a 7-DoF action through the public adapter on a "
            "local RTX 4070 Ti SUPER (4-bit), with measured load time, VRAM, and latency.",
            (_link("local runtime evidence", "../openvla_local_runtime.md"),),
        ),
        "gpu_inference": _cell(
            "verified",
            "4-bit (nf4) loading fits a 16 GB consumer GPU: ~4.6 GB peak VRAM, ~1.1-2.7 s per "
            "inference. Measured via scripts/measure_openvla_runtime.py.",
            (_link("local runtime evidence", "../openvla_local_runtime.md"),),
        ),
        "remote_server": _cell(
            "verified",
            "A real `vla-zoo serve --model openvla --load-in-4bit` server passed a "
            "health-first probe and returned a typed 7-DoF action over HTTP /v1/predict "
            "(recorded end-to-end on a 16 GB GPU).",
            (
                _link(
                    "OpenVLA remote probe",
                    "sample_task_verification/openvla_remote_probe.md",
                ),
                _link("OpenVLA remote path", "../openvla_remote.md"),
            ),
        ),
        "ros2_remote": _cell(
            "planned",
            "ROS2 remote command is generated; checked-in OpenVLA ROS2 action logs are "
            "still needed.",
            (_link("ROS2 remote plan", "ros2_remote_smoke_plan.md"),),
        ),
        "pybullet_tasks": _cell(
            "planned",
            "PyBullet runner can call OpenVLA locally or remotely; heavy local run is "
            "skipped by default.",
            (
                _link(
                    "external adapter status",
                    "sample_task_verification/external_adapter_status.md",
                ),
            ),
        ),
        "policy_quality": _cell(
            "not_verified",
            "No task-success or robot-skill claim is made for OpenVLA in this repository.",
        ),
    }


def _pi0_evidence(info: AdapterInfo) -> dict[str, EvidenceCell]:
    return {
        "contract": _registry_contract(info),
        "local_runtime": _cell(
            "blocked",
            "Local load fails on a concrete config-schema mismatch: the cached lerobot/pi0 "
            "checkpoint carries PI0Config fields (resize_imgs_with_padding, adapt_to_pi_aloha, "
            "num_steps, ...) that LeRobot 0.5.1 rejects. Needs a version-matched checkpoint.",
            (
                _link(
                    "pi0 compatibility note",
                    "sample_task_verification/pi0_compatibility_probe.md",
                ),
            ),
        ),
        "gpu_inference": _cell(
            "planned",
            "Needs a dedicated openpi or LeRobot serving environment and a recorded action probe.",
            (_link("adapter card", "../adapters/pi0.md"),),
        ),
        "remote_server": _cell(
            "planned",
            "Remote-first deployment path with a reproducible pi0 server plan and "
            "LeRobot/openpi version-compatibility docs; a recorded pi0 /v1/predict run "
            "from a version-matched server is still needed.",
            (
                _link("pi0 remote path", "../pi0_remote.md"),
                _link("pi0 server plan", "pi0_server_plan.md"),
            ),
        ),
        "ros2_remote": _cell(
            "planned",
            "ROS2 remote launch can target a pi0 server; checked-in action logs are still needed.",
            (_link("ROS2 remote plan", "ros2_remote_smoke_plan.md"),),
        ),
        "pybullet_tasks": _cell(
            "planned",
            "Runner has the observation plumbing; real pi0 remote task traces are not checked in.",
            (
                _link(
                    "external adapter status",
                    "sample_task_verification/external_adapter_status.md",
                ),
            ),
        ),
        "policy_quality": _cell(
            "not_verified",
            "No task-success or robot-skill claim is made for pi0/openpi in this repository.",
        ),
    }


def _smolvla_evidence(info: AdapterInfo) -> dict[str, EvidenceCell]:
    return {
        "contract": _registry_contract(info),
        "local_runtime": _cell(
            "verified",
            "lerobot/smolvla_base loaded and predicted a 6-DoF action through "
            "load_model('smolvla') on a local RTX 4070 Ti SUPER, with measured load/VRAM/latency.",
            (_link("local runtime evidence", "../smolvla_local_runtime.md"),),
        ),
        "gpu_inference": _cell(
            "verified",
            "CUDA inference measured at ~0.97 GB peak VRAM and ~60-133 ms steady latency "
            "(real-time capable). Captured via scripts/measure_lerobot_runtime.py.",
            (_link("local runtime evidence", "../smolvla_local_runtime.md"),),
        ),
        "remote_server": _cell(
            "verified",
            "A real `vla-zoo serve --model smolvla` server passed a health-first probe and "
            "returned a typed 6-DoF action over HTTP /v1/predict (recorded end-to-end).",
            (
                _link(
                    "SmolVLA remote probe",
                    "sample_task_verification/smolvla_remote_probe.md",
                ),
                _link("SmolVLA remote plan", "smolvla_remote_smoke_plan.md"),
            ),
        ),
        "ros2_remote": _cell(
            "verified",
            "The real VLARuntimeNode ran in remote mode against a live SmolVLA server: "
            "recorded 14 RemoteVLAClient actions + 106 status/diagnostics with 0 inference "
            "errors (vla-zoo ros remote-smoke-check passed).",
            (
                _link(
                    "ROS2 remote smoke check",
                    "sample_ros2_remote_smolvla/remote_smoke_check.md",
                ),
                _link("ROS2 remote plan", "ros2_remote_smoke_plan.md"),
            ),
        ),
        "pybullet_tasks": _cell(
            "partial",
            "Rendered multi-camera plus state observation path was probed; not a "
            "task-success benchmark.",
            (
                _link(
                    "SmolVLA PyBullet report",
                    "sample_task_verification/smolvla_pybullet_report.html",
                ),
            ),
        ),
        "policy_quality": _cell(
            "not_verified",
            "SmolVLA base still needs robot/task-specific fine-tuning and calibration.",
        ),
    }


def _groot_evidence(info: AdapterInfo) -> dict[str, EvidenceCell]:
    return {
        "contract": _cell(
            "partial",
            "Runtime contract is declared, but GR00T is blocked until the NVIDIA Isaac GR00T "
            "stack is wired in; no inference is implemented.",
            (
                _link("adapter card", "../adapters/groot.md"),
                _link("blocked status", "../groot_remote.md"),
            ),
        ),
        "local_runtime": _cell(
            "blocked",
            "Blocked until the NVIDIA Isaac GR00T stack is wired in; no GR00T inference ships "
            "and the adapter raises instead of fabricating actions.",
            (
                _link("blocked status", "../groot_remote.md"),
                _link(
                    "external adapter status",
                    "sample_task_verification/external_adapter_status.md",
                ),
            ),
        ),
        "gpu_inference": _cell(
            "blocked",
            "Requires the dedicated NVIDIA Isaac GR00T stack and a recorded action probe; "
            "blocked until a real serving adapter exists.",
            (_link("blocked status", "../groot_remote.md"),),
        ),
        "remote_server": _cell(
            "blocked",
            "Expected to run through a remote serving environment once a real GR00T serving "
            "adapter lands; blocked until then.",
            (
                _link("blocked status", "../groot_remote.md"),
                _link("GPU server plan", "sample_compare_suite/gpu_server_plan.md"),
            ),
        ),
        "ros2_remote": _cell(
            "planned",
            "ROS2 remote launch can target a GR00T server after real serving support exists.",
            (_link("ROS2 remote plan", "ros2_remote_smoke_plan.md"),),
        ),
        "pybullet_tasks": _cell(
            "not_verified",
            "No real GR00T action traces are checked in.",
        ),
        "policy_quality": _cell(
            "not_verified",
            "No task-success or robot-skill claim is made for GR00T in this repository.",
        ),
    }


def _generic_evidence(info: AdapterInfo) -> dict[str, EvidenceCell]:
    return {
        "contract": _registry_contract(info),
        "local_runtime": _cell(
            "not_verified",
            "No checked-in runtime evidence is registered for this adapter yet.",
        ),
        "gpu_inference": _cell(
            "not_verified",
            "No checked-in GPU inference evidence is registered for this adapter yet.",
        ),
        "remote_server": _cell(
            "planned",
            "External adapters can use vla_zoo remote serving once implemented.",
        ),
        "ros2_remote": _cell(
            "planned",
            "External adapters can use the ROS2 remote runtime once a server is available.",
        ),
        "pybullet_tasks": _cell(
            "not_verified",
            "No checked-in PyBullet traces are registered for this adapter yet.",
        ),
        "policy_quality": _cell(
            "not_verified",
            "No task-success or robot-skill claim is made for this adapter.",
        ),
    }


def _evidence_for(info: AdapterInfo) -> dict[str, EvidenceCell]:
    if info.metadata.get("baseline"):
        return _baseline_evidence(info)
    if info.name == "dummy":
        return _baseline_evidence(info)
    if info.name == "openvla":
        return _openvla_evidence(info)
    if info.name == "pi0":
        return _pi0_evidence(info)
    if info.name == "smolvla":
        return _smolvla_evidence(info)
    if info.name == "groot":
        return _groot_evidence(info)
    return _generic_evidence(info)


def _next_step(info: AdapterInfo) -> str:
    if info.name == "dummy":
        return "Keep this as the CI and ROS2 remote wiring baseline."
    if info.metadata.get("baseline"):
        return "Keep using this baseline for simulation/report regression checks."
    if info.name == "openvla":
        return (
            "Local 4-bit GPU inference and a real remote /v1/predict are both verified; "
            "next, record a ROS2 remote action log, then add task-level probes."
        )
    if info.name == "pi0":
        return (
            "Stand up a dedicated pi0/openpi server and record /v1/predict plus ROS2 "
            "remote logs."
        )
    if info.name == "smolvla":
        return (
            "Local GPU, remote /v1/predict, and a ROS2 remote trace are all verified; "
            "next, broaden to task-level probes on real scene frames."
        )
    if info.name == "groot":
        return (
            "Replace the placeholder with a real GR00T serving adapter before claiming "
            "inference."
        )
    return "Add adapter-specific evidence links and runtime checks."


def build_evidence_matrix(
    models: Sequence[str],
    *,
    status_provider: Callable[[str], str] | None = None,
) -> list[ModelEvidence]:
    records: list[ModelEvidence] = []
    seen: set[str] = set()
    for model in models:
        info = get_adapter_info(model)
        if info.name in seen:
            continue
        seen.add(info.name)
        metadata = info.metadata
        records.append(
            ModelEvidence(
                model=info.name,
                upstream_project=str(metadata.get("upstream_project", "-")),
                family=str(metadata.get("family", "adapter")),
                adapter_status=status_provider(info.name) if status_provider else "unknown",
                experimental=info.experimental,
                evidence=_evidence_for(info),
                next_step=_next_step(info),
                caveat=str(
                    metadata.get(
                        "verification",
                        "No adapter-specific verification note has been declared yet.",
                    )
                ),
            )
        )
    return records


def evidence_matrix_payload(records: Sequence[ModelEvidence]) -> dict[str, object]:
    return {
        "schema": "vla_zoo.vla_model_evidence_matrix.v1",
        "columns": list(EVIDENCE_COLUMNS),
        "records": [record.to_dict() for record in records],
        "scope": (
            "Runtime evidence matrix. This does not claim robot task success unless a cell "
            "explicitly points to a task-success benchmark."
        ),
    }


def _table_value(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


def _cell_summary(cell: EvidenceCell) -> str:
    return f"{cell.status}: {cell.summary}"


def _links_markdown(links: Sequence[EvidenceLink]) -> str:
    if not links:
        return "-"
    return ", ".join(f"[{link.label}]({link.href})" for link in links)


def _links_html(links: Sequence[EvidenceLink]) -> str:
    if not links:
        return '<span class="muted">No checked-in artifact yet</span>'
    return " ".join(
        f'<a href="{escape(link.href, quote=True)}">{escape(link.label)}</a>' for link in links
    )


def _status_counts(records: Sequence[ModelEvidence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for cell in record.evidence.values():
            counts[cell.status] = counts.get(cell.status, 0) + 1
    return counts


def format_evidence_matrix_markdown(
    records: Sequence[ModelEvidence],
    *,
    title: str = "VLA Model Evidence Matrix",
) -> str:
    lines = [
        f"# {title}",
        "",
        "This matrix records what has actually been exercised through `vla_zoo` and what is "
        "still only planned. It is intentionally runtime-centric.",
        "",
        "It is not a model-quality leaderboard. `verified` means the repository contains a "
        "checked-in runtime artifact or deterministic test for that cell.",
        "",
        "| Model | Family | Adapter status | Contract | GPU inference | Remote server | "
        "ROS2 remote | PyBullet tasks | Policy quality |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        cells = record.evidence
        lines.append(
            f"| `{record.model}` | {_table_value(record.family)} | "
            f"{_table_value(record.adapter_status)} | "
            f"{_table_value(_cell_summary(cells['contract']))} | "
            f"{_table_value(_cell_summary(cells['gpu_inference']))} | "
            f"{_table_value(_cell_summary(cells['remote_server']))} | "
            f"{_table_value(_cell_summary(cells['ros2_remote']))} | "
            f"{_table_value(_cell_summary(cells['pybullet_tasks']))} | "
            f"{_table_value(_cell_summary(cells['policy_quality']))} |"
        )

    lines.extend(
        [
            "",
            "## Evidence Links",
            "",
        ]
    )
    for record in records:
        lines.extend(
            [
                f"### {record.model}",
                "",
                f"- Upstream: {record.upstream_project}",
                f"- Next step: {record.next_step}",
                f"- Caveat: {record.caveat}",
                "",
                "| Cell | Status | Evidence |",
                "|---|---|---|",
            ]
        )
        for key in EVIDENCE_COLUMNS:
            cell = record.evidence[key]
            label = COLUMN_LABELS[key]
            lines.append(
                f"| {label} | {cell.status} | "
                f"{_table_value(cell.summary)}<br>{_links_markdown(cell.links)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Reading Rules",
            "",
            "- `verified`: checked-in tests, logs, or reports exist for the runtime path.",
            "- `partial`: the adapter path exists, but the evidence is incomplete for that cell.",
            "- `planned`: commands or scaffolding exist, but a checked-in run is still missing.",
            "- `blocked`: the current repo run could not complete due to dependency, "
            "memory, or stack limits.",
            "- `not_verified`: no claim is made.",
            "- `not_applicable`: the cell does not apply to that baseline or adapter.",
            "",
            "For real robots, this matrix must be paired with action clipping, stale-action "
            "watchdogs, emergency stop integration, and a hardware-specific bridge.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_evidence_matrix_html(
    records: Sequence[ModelEvidence],
    *,
    title: str = "VLA Model Evidence Matrix",
) -> str:
    """Render a standalone Pages-friendly evidence report."""

    counts = _status_counts(records)
    status_order: tuple[EvidenceStatus, ...] = (
        "verified",
        "partial",
        "planned",
        "blocked",
        "not_verified",
        "not_applicable",
    )
    summary_cards = "\n".join(
        (
            f'<div class="summary-card status-{status}">'
            f"<span>{escape(status.replace('_', ' '))}</span>"
            f"<strong>{counts.get(status, 0)}</strong>"
            "</div>"
        )
        for status in status_order
    )
    model_cards = []
    for record in records:
        cell_rows = []
        for key in EVIDENCE_COLUMNS:
            cell = record.evidence[key]
            cell_rows.append(
                "<tr>"
                f"<th>{escape(COLUMN_LABELS[key])}</th>"
                f'<td><span class="badge status-{cell.status}">'
                f"{escape(cell.status.replace('_', ' '))}</span></td>"
                f"<td>{escape(cell.summary)}</td>"
                f"<td>{_links_html(cell.links)}</td>"
                "</tr>"
            )
        model_cards.append(
            '<section class="model-card">'
            '<div class="model-head">'
            "<div>"
            f"<h2>{escape(record.model)}</h2>"
            f"<p>{escape(record.family)} · {escape(record.upstream_project)}</p>"
            "</div>"
            f'<span class="adapter-status">{escape(record.adapter_status)}</span>'
            "</div>"
            '<div class="model-meta">'
            f"<p><strong>Next step</strong><br>{escape(record.next_step)}</p>"
            f"<p><strong>Caveat</strong><br>{escape(record.caveat)}</p>"
            "</div>"
            '<table class="evidence-table">'
            "<thead><tr><th>Cell</th><th>Status</th><th>Meaning</th><th>Artifact</th></tr></thead>"
            f"<tbody>{''.join(cell_rows)}</tbody>"
            "</table>"
            "</section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #64748b;
      --line: #dbe3ef;
      --panel: #ffffff;
      --bg: #f6f8fb;
      --verified: #0f766e;
      --partial: #b45309;
      --planned: #2563eb;
      --blocked: #b91c1c;
      --not-verified: #475569;
      --not-applicable: #6b7280;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    header {{
      background: #0f172a;
      color: white;
      padding: 34px 24px 28px;
    }}
    header .wrap, main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 4vw, 46px);
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0;
      font-size: 24px;
      letter-spacing: 0;
    }}
    p {{ margin: 0; }}
    a {{ color: #0369a1; font-weight: 700; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .lede {{
      max-width: 820px;
      color: #cbd5e1;
      font-size: 17px;
    }}
    main {{ padding: 22px 0 42px; }}
    .notice {{
      border: 1px solid #facc15;
      background: #fefce8;
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 18px;
      color: #713f12;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin: 0 0 20px;
    }}
    .summary-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-left-width: 5px;
      border-radius: 8px;
      padding: 12px;
    }}
    .summary-card span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
    }}
    .summary-card strong {{
      display: block;
      margin-top: 4px;
      font-size: 30px;
    }}
    .model-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-top: 16px;
      overflow: hidden;
    }}
    .model-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      padding: 18px;
      border-bottom: 1px solid var(--line);
    }}
    .model-head p, .muted {{
      color: var(--muted);
    }}
    .adapter-status {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    .model-meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
      padding: 14px 18px;
      background: #f8fafc;
      border-bottom: 1px solid var(--line);
    }}
    .model-meta p {{
      color: #334155;
      font-size: 14px;
    }}
    .evidence-table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{
      width: 150px;
      color: #334155;
      font-size: 14px;
    }}
    thead th {{
      background: #eef4f8;
      color: #0f172a;
    }}
    td:nth-child(2) {{ width: 135px; }}
    td:nth-child(4) {{ width: 190px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      border-radius: 999px;
      padding: 4px 9px;
      color: white;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }}
    .status-verified {{ border-left-color: var(--verified); }}
    .status-partial {{ border-left-color: var(--partial); }}
    .status-planned {{ border-left-color: var(--planned); }}
    .status-blocked {{ border-left-color: var(--blocked); }}
    .status-not_verified {{ border-left-color: var(--not-verified); }}
    .status-not_applicable {{ border-left-color: var(--not-applicable); }}
    .badge.status-verified {{ background: var(--verified); }}
    .badge.status-partial {{ background: var(--partial); }}
    .badge.status-planned {{ background: var(--planned); }}
    .badge.status-blocked {{ background: var(--blocked); }}
    .badge.status-not_verified {{ background: var(--not-verified); }}
    .badge.status-not_applicable {{ background: var(--not-applicable); }}
    .reading-rules {{
      margin-top: 20px;
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 760px) {{
      .model-head {{ display: block; }}
      .adapter-status {{ display: inline-block; margin-top: 10px; }}
      .evidence-table, thead, tbody, tr, th, td {{ display: block; width: 100%; }}
      thead {{ display: none; }}
      tr {{ border-bottom: 1px solid var(--line); }}
      th, td {{ border-bottom: 0; padding: 8px 14px; }}
      th {{ padding-top: 14px; }}
      td:nth-child(2), td:nth-child(4) {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>{escape(title)}</h1>
      <p class="lede">
        Runtime evidence for VLA adapter contracts, GPU inference, remote serving,
        ROS2 remote paths, PyBullet traces, and policy-quality claims.
      </p>
    </div>
  </header>
  <main>
    <div class="notice">
      This is not a model-quality leaderboard. Verified means this repository contains
      a checked-in runtime artifact, deterministic test, or recorded report for that cell.
    </div>
    <section class="summary" aria-label="Status summary">
      {summary_cards}
    </section>
    {''.join(model_cards)}
    <p class="reading-rules">
      Reading rules: partial means evidence is incomplete; planned means scaffolding or
      commands exist but a checked-in run is missing; blocked means dependency, memory,
      or stack limits prevented completion; not verified means no claim is made.
    </p>
  </main>
</body>
</html>
"""
