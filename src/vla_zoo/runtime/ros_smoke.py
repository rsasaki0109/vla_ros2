from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class RosRemoteSmokeCheck:
    ok: bool
    expected_model: str
    expected_remote_url: str
    status_count: int
    action_count: int
    diagnostics_count: int
    ready_count: int
    dry_run_count: int
    remote_status_count: int
    remote_diagnostics_count: int
    remote_action_count: int
    inference_error_count: int
    diagnostic_error_count: int
    model_names: tuple[str, ...]
    adapter_names: tuple[str, ...]
    action_spaces: tuple[str, ...]
    mean_latency_ms: float | None
    max_latency_ms: float | None
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema"] = "vla_zoo.ros_remote_smoke_check.v1"
        return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata")
    if isinstance(raw, dict):
        return dict(raw)
    text = row.get("metadata_json")
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _diagnostic_values(row: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    statuses = row.get("status")
    if not isinstance(statuses, list):
        return values
    for status in statuses:
        if not isinstance(status, dict):
            continue
        raw_values = status.get("values")
        if not isinstance(raw_values, list):
            continue
        for item in raw_values:
            if isinstance(item, dict):
                key = item.get("key")
                value = item.get("value")
                if isinstance(key, str):
                    values[key] = str(value)
    return values


def _diagnostic_error_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        statuses = row.get("status")
        if not isinstance(statuses, list):
            continue
        for status in statuses:
            if isinstance(status, dict) and int(status.get("level", 0)) >= 2:
                count += 1
    return count


def _float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def check_ros_remote_smoke(
    *,
    action_log: Path,
    status_log: Path,
    diagnostics_log: Path,
    expected_model: str,
    expected_remote_url: str,
    require_actions: bool = True,
    require_diagnostics: bool = True,
) -> RosRemoteSmokeCheck:
    action_rows = _read_jsonl(action_log)
    status_rows = _read_jsonl(status_log)
    diagnostic_rows = _read_jsonl(diagnostics_log)

    model_names = sorted(
        {
            str(row.get("model_name"))
            for row in (*status_rows, *action_rows)
            if row.get("model_name")
        }
    )
    adapter_names = sorted(
        {
            str(row.get("adapter_name"))
            for row in (*status_rows, *action_rows)
            if row.get("adapter_name")
        }
    )
    action_spaces = sorted(
        {str(row.get("action_space")) for row in action_rows if row.get("action_space")}
    )
    remote_status_count = sum(
        1
        for row in status_rows
        if _metadata(row).get("runtime") == "remote"
        and _metadata(row).get("remote_url") == expected_remote_url
    )
    remote_action_count = sum(
        1
        for row in action_rows
        if row.get("model_name") == expected_model
        and str(row.get("adapter_name")) == "RemoteVLAClient"
    )
    remote_diagnostics_count = sum(
        1
        for row in diagnostic_rows
        if _diagnostic_values(row).get("runtime") == "remote"
        and _diagnostic_values(row).get("remote_url") == expected_remote_url
    )
    latency_values = _float_values(status_rows, "last_latency_ms")
    inference_error_count = sum(
        1 for row in status_rows if str(row.get("status_text", "")).startswith("inference error")
    )
    diagnostic_error_count = _diagnostic_error_count(diagnostic_rows)

    issues: list[str] = []
    if not status_rows:
        issues.append(f"missing status records: {status_log}")
    if require_actions and not action_rows:
        issues.append(f"missing action records: {action_log}")
    if require_diagnostics and not diagnostic_rows:
        issues.append(f"missing diagnostics records: {diagnostics_log}")
    if expected_model not in model_names:
        issues.append(f"expected model {expected_model!r} was not recorded")
    if remote_status_count == 0:
        issues.append(
            f"no status record declared runtime=remote and remote_url={expected_remote_url!r}"
        )
    if require_actions and remote_action_count == 0:
        issues.append("no VLAAction record came from RemoteVLAClient")
    if require_diagnostics and remote_diagnostics_count == 0:
        issues.append(
            f"no diagnostic record declared runtime=remote and remote_url={expected_remote_url!r}"
        )
    if inference_error_count:
        issues.append(f"inference error status records: {inference_error_count}")
    if diagnostic_error_count:
        issues.append(f"diagnostic error records: {diagnostic_error_count}")
    if not any(bool(row.get("dry_run")) for row in status_rows):
        issues.append("no status record confirmed dry_run=true")

    return RosRemoteSmokeCheck(
        ok=not issues,
        expected_model=expected_model,
        expected_remote_url=expected_remote_url,
        status_count=len(status_rows),
        action_count=len(action_rows),
        diagnostics_count=len(diagnostic_rows),
        ready_count=sum(1 for row in status_rows if bool(row.get("ready"))),
        dry_run_count=sum(1 for row in status_rows if bool(row.get("dry_run"))),
        remote_status_count=remote_status_count,
        remote_diagnostics_count=remote_diagnostics_count,
        remote_action_count=remote_action_count,
        inference_error_count=inference_error_count,
        diagnostic_error_count=diagnostic_error_count,
        model_names=tuple(model_names),
        adapter_names=tuple(adapter_names),
        action_spaces=tuple(action_spaces),
        mean_latency_ms=mean(latency_values) if latency_values else None,
        max_latency_ms=max(latency_values) if latency_values else None,
        issues=tuple(issues),
    )


def format_ros_remote_smoke_check_markdown(
    check: RosRemoteSmokeCheck,
    *,
    title: str = "ROS2 Remote Runtime Smoke Check",
) -> str:
    def fmt(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}"

    lines = [
        f"# {title}",
        "",
        "This report validates the ROS2 remote runtime path from recorded JSONL logs:",
        "`vla_runtime_node` status, diagnostics, and published `VLAAction` messages.",
        "",
        "It proves transport and runtime wiring, not VLA policy quality or hardware safety.",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| overall | {'ok' if check.ok else 'check'} |",
        f"| status_count | {check.status_count} |",
        f"| action_count | {check.action_count} |",
        f"| diagnostics_count | {check.diagnostics_count} |",
        f"| ready_count | {check.ready_count} |",
        f"| dry_run_count | {check.dry_run_count} |",
        f"| remote_status_count | {check.remote_status_count} |",
        f"| remote_action_count | {check.remote_action_count} |",
        f"| remote_diagnostics_count | {check.remote_diagnostics_count} |",
        f"| inference_error_count | {check.inference_error_count} |",
        f"| diagnostic_error_count | {check.diagnostic_error_count} |",
        f"| mean_latency_ms | {fmt(check.mean_latency_ms)} |",
        f"| max_latency_ms | {fmt(check.max_latency_ms)} |",
        "",
        "## Runtime Evidence",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| expected_model | `{check.expected_model}` |",
        f"| expected_remote_url | `{check.expected_remote_url}` |",
        f"| models_seen | {', '.join(f'`{name}`' for name in check.model_names) or '-'} |",
        f"| adapters_seen | {', '.join(f'`{name}`' for name in check.adapter_names) or '-'} |",
        f"| action_spaces | {', '.join(f'`{name}`' for name in check.action_spaces) or '-'} |",
        "",
        "## Scope",
        "",
        "- `remote_status_count` means status metadata reported `runtime=remote` "
        "and the expected URL.",
        "- `remote_action_count` means recorded actions came from `RemoteVLAClient`.",
        "- `remote_diagnostics_count` means diagnostics reported the remote runtime and URL.",
        "- `dry_run=true` keeps this path on the typed-action publication boundary.",
        "- This does not command hardware and does not validate model task success.",
    ]
    if check.issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in check.issues)
    return "\n".join(lines) + "\n"
