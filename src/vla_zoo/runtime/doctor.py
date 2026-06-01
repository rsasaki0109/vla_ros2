from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from importlib.util import find_spec
from typing import Any, Literal

from vla_zoo.core.registry import get_adapter_info, list_models, load_model
from vla_zoo.core.types import VLAActionChunk

Severity = Literal["ok", "warn", "error"]


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.severity == "ok"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def _module_available(name: str) -> bool:
    return find_spec(name) is not None


def _command_available(name: str) -> bool:
    return shutil.which(name) is not None


def _check_python() -> DoctorCheck:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return DoctorCheck("python", "ok", f"Python {version}", {"version": version})


def _check_module(name: str, import_name: str, *, required: bool) -> DoctorCheck:
    available = _module_available(import_name)
    severity: Severity = "ok" if available else ("error" if required else "warn")
    kind = "required" if required else "optional"
    message = f"{kind} module {import_name} {'available' if available else 'missing'}"
    return DoctorCheck(f"module.{name}", severity, message, {"import_name": import_name})


def _check_command(name: str, *, required: bool = False) -> DoctorCheck:
    path = shutil.which(name)
    severity: Severity = "ok" if path else ("error" if required else "warn")
    message = f"command {name} {'available' if path else 'not found'}"
    return DoctorCheck(f"command.{name}", severity, message, {"path": path})


def _check_dummy_predict() -> DoctorCheck:
    try:
        model = load_model("dummy")
        action = model.predict(image=None, instruction="doctor")
    except Exception as exc:
        return DoctorCheck("adapter.dummy.predict", "error", str(exc))
    spec = action.actions[0].spec if isinstance(action, VLAActionChunk) else action.spec
    return DoctorCheck(
        "adapter.dummy.predict",
        "ok",
        f"dummy prediction returned {spec.action_space}{spec.shape}",
    )


def _check_adapter(name: str) -> DoctorCheck:
    info = get_adapter_info(name)
    if name == "dummy":
        return DoctorCheck("adapter.dummy", "ok", "always-available dry-run adapter")
    if name == "openvla":
        missing = [
            import_name
            for import_name in ("torch", "transformers")
            if not _module_available(import_name)
        ]
        if missing:
            return DoctorCheck(
                "adapter.openvla",
                "warn",
                'missing optional deps; install with pip install "vla_zoo[openvla]"',
                {"missing": missing},
            )
        return DoctorCheck("adapter.openvla", "ok", "OpenVLA optional deps available")
    severity: Severity = "warn" if info.experimental else "ok"
    message = "experimental adapter scaffold" if info.experimental else "adapter registered"
    return DoctorCheck(
        f"adapter.{info.name}",
        severity,
        message,
        {"aliases": list(info.aliases), "domain": info.domain},
    )


def _check_remote(remote_url: str) -> DoctorCheck:
    if not _module_available("httpx"):
        return DoctorCheck(
            "remote.health",
            "error",
            'httpx missing; install with pip install "vla_zoo[server]"',
            {"remote_url": remote_url},
        )
    import httpx

    try:
        response = httpx.get(f"{remote_url.rstrip('/')}/health", timeout=2.0)
    except Exception as exc:
        return DoctorCheck(
            "remote.health",
            "error",
            f"could not reach {remote_url}: {exc}",
            {"remote_url": remote_url},
        )
    severity: Severity = "ok" if response.status_code < 400 else "error"
    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = {"text": response.text[:200]}
    return DoctorCheck(
        "remote.health",
        severity,
        f"GET /health returned HTTP {response.status_code}",
        {"remote_url": remote_url, "response": payload},
    )


def run_doctor(
    *,
    include_ros: bool = True,
    remote_url: str | None = None,
) -> list[DoctorCheck]:
    checks = [
        _check_python(),
        _check_module("numpy", "numpy", required=True),
        _check_module("pydantic", "pydantic", required=True),
        _check_module("pillow", "PIL", required=True),
        _check_module("typer", "typer", required=False),
        _check_module("fastapi", "fastapi", required=False),
        _check_module("httpx", "httpx", required=False),
        _check_dummy_predict(),
    ]
    checks.extend(_check_adapter(adapter.name) for adapter in list_models())
    if include_ros:
        checks.extend([_check_command("ros2"), _check_command("colcon")])
    if remote_url:
        checks.append(_check_remote(remote_url))
    return checks


def summarize_checks(checks: list[DoctorCheck]) -> dict[str, int]:
    return {
        "ok": sum(1 for check in checks if check.severity == "ok"),
        "warn": sum(1 for check in checks if check.severity == "warn"),
        "error": sum(1 for check in checks if check.severity == "error"),
    }


def format_doctor_table(checks: list[DoctorCheck]) -> str:
    lines = [f"{'status':<7} {'check':<28} message", f"{'-' * 7} {'-' * 28} {'-' * 40}"]
    for check in checks:
        lines.append(f"{check.severity:<7} {check.name:<28} {check.message}")
    summary = summarize_checks(checks)
    lines.append("")
    lines.append(
        f"summary: {summary['ok']} ok, {summary['warn']} warn, {summary['error']} error"
    )
    return "\n".join(lines)
