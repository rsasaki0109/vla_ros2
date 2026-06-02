from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
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
            "partial",
            "Lazy Hugging Face adapter and prompt path are implemented without test-time "
            "downloads.",
            (_link("adapter card", "../adapters/openvla.md"),),
        ),
        "gpu_inference": _cell(
            "blocked",
            "Local CUDA prompt probe found weights/deps but did not complete due to free VRAM.",
            (_link("OpenVLA prompt probe", "sample_task_verification/openvla_prompt_probe.md"),),
        ),
        "remote_server": _cell(
            "planned",
            "GPU server command is generated; recorded OpenVLA /v1/predict run is still needed.",
            (_link("GPU server plan", "sample_compare_suite/gpu_server_plan.md"),),
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
            "Local inference is opt-in because checkpoint/config compatibility varies by "
            "LeRobot version.",
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
            "Remote-first deployment path is the intended initial route for pi0.",
            (_link("GPU server plan", "sample_compare_suite/gpu_server_plan.md"),),
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
            "LeRobot SmolVLA adapter returned a typed action through load_model('smolvla').",
            (_link("SmolVLA GPU probe", "sample_task_verification/smolvla_gpu_probe.md"),),
        ),
        "gpu_inference": _cell(
            "verified",
            "CUDA inference-path probe completed with lerobot/smolvla_base.",
            (_link("SmolVLA GPU probe", "sample_task_verification/smolvla_gpu_probe.md"),),
        ),
        "remote_server": _cell(
            "planned",
            "Server command can be generated; recorded SmolVLA /v1/predict run is still needed.",
            (_link("GPU server plan", "sample_compare_suite/gpu_server_plan.md"),),
        ),
        "ros2_remote": _cell(
            "planned",
            "ROS2 remote launch can target a SmolVLA server; checked-in action logs are "
            "still needed.",
            (_link("ROS2 remote plan", "ros2_remote_smoke_plan.md"),),
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
            "Experimental adapter target is declared, but real GR00T inference is not implemented.",
            (_link("adapter card", "../adapters/groot.md"),),
        ),
        "local_runtime": _cell(
            "blocked",
            "Isaac GR00T dependencies are not installed or verified in this repository.",
            (
                _link(
                    "external adapter status",
                    "sample_task_verification/external_adapter_status.md",
                ),
            ),
        ),
        "gpu_inference": _cell(
            "planned",
            "Requires a dedicated NVIDIA GR00T stack and a recorded action probe.",
        ),
        "remote_server": _cell(
            "planned",
            "Expected to run through a remote serving environment once adapter support lands.",
            (_link("GPU server plan", "sample_compare_suite/gpu_server_plan.md"),),
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
            "Run OpenVLA behind a GPU server with enough free VRAM and check in remote "
            "action logs."
        )
    if info.name == "pi0":
        return (
            "Stand up a dedicated pi0/openpi server and record /v1/predict plus ROS2 "
            "remote logs."
        )
    if info.name == "smolvla":
        return "Record SmolVLA remote-server and ROS2 remote traces, then broaden task probes."
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
