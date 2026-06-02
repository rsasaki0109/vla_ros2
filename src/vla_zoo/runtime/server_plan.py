from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from shlex import quote

DEFAULT_REMOTE_MODELS = ("openvla", "pi0", "smolvla", "groot")


@dataclass(frozen=True)
class ServerPlanEntry:
    model: str
    port: int
    host: str
    public_url: str
    install_hint: str
    command: tuple[str, ...]
    notes: str

    def shell_command(self) -> str:
        return " ".join(quote(part) for part in self.command)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        payload["shell_command"] = self.shell_command()
        return payload


@dataclass(frozen=True)
class ServerPlan:
    entries: tuple[ServerPlanEntry, ...]
    public_host: str
    runtime: str = "remote"

    @property
    def remote_map(self) -> str:
        return ",".join(f"{entry.model}={entry.public_url}" for entry in self.entries)

    @property
    def compare_command(self) -> str:
        models = ",".join(entry.model for entry in self.entries)
        return (
            "vla-zoo compare pybullet "
            f"--models {quote(models)} "
            "--runtime remote "
            f"--remote-map {quote(self.remote_map)}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime": self.runtime,
            "public_host": self.public_host,
            "remote_map": self.remote_map,
            "compare_command": self.compare_command,
            "servers": [entry.to_dict() for entry in self.entries],
        }


def _default_pretrained(model: str) -> str | None:
    if model == "openvla":
        return "openvla/openvla-7b"
    if model == "pi0":
        return "lerobot/pi0_base"
    if model == "smolvla":
        return "lerobot/smolvla_base"
    return None


def _install_hint(model: str) -> str:
    if model == "openvla":
        return 'pip install -e ".[cli,server,openvla]"'
    if model == "pi0":
        return 'pip install -e ".[cli,server,openpi]"'
    if model == "smolvla":
        return 'pip install -e ".[cli,server,smolvla]"'
    if model == "groot":
        return 'Install the external GR00T stack, then pip install -e ".[cli,server,groot]"'
    return 'pip install -e ".[cli,server]"'


def _notes(model: str) -> str:
    if model == "openvla":
        return "Requires external OpenVLA weights and enough GPU memory."
    if model == "pi0":
        return (
            "Uses explicit LeRobot checkpoint selection; checkpoint compatibility is "
            "version-sensitive."
        )
    if model == "smolvla":
        return "Uses LeRobot policy loading with multi-camera/state observations."
    if model == "groot":
        return (
            "Blocked until the NVIDIA Isaac GR00T stack is wired in; this plan is a placeholder "
            "and ships no inference."
        )
    return "Lightweight runtime server."


def _normalize_models(models: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(model.strip().lower() for model in models if model.strip())
    if not normalized:
        msg = "At least one model is required for a server plan."
        raise ValueError(msg)
    return normalized


def build_server_plan(
    models: Sequence[str] = DEFAULT_REMOTE_MODELS,
    *,
    host: str = "0.0.0.0",
    public_host: str = "gpu-box",
    base_port: int = 8001,
    device: str = "cuda:0",
    dtype: str | None = None,
    unnorm_key: str | None = "bridge_orig",
    pretrained: Mapping[str, str] | None = None,
) -> ServerPlan:
    """Build a multi-server plan for remote VLA comparison."""

    pretrained = {key.lower(): value for key, value in (pretrained or {}).items()}
    entries: list[ServerPlanEntry] = []
    for index, model in enumerate(_normalize_models(models)):
        port = base_port + index
        command = ["vla-zoo", "serve", "--model", model, "--host", host, "--port", str(port)]
        checkpoint = pretrained.get(model, _default_pretrained(model))
        if checkpoint:
            command.extend(["--pretrained", checkpoint])
        if model in {"openvla", "pi0", "smolvla"}:
            command.extend(["--device", device])
        if dtype and model in {"openvla", "smolvla"}:
            command.extend(["--dtype", dtype])
        if unnorm_key and model == "openvla":
            command.extend(["--unnorm-key", unnorm_key])
        entries.append(
            ServerPlanEntry(
                model=model,
                port=port,
                host=host,
                public_url=f"http://{public_host}:{port}",
                install_hint=_install_hint(model),
                command=tuple(command),
                notes=_notes(model),
            )
        )
    return ServerPlan(entries=tuple(entries), public_host=public_host)


def format_server_plan_markdown(plan: ServerPlan) -> str:
    lines = [
        "# VLA GPU Server Plan",
        "",
        "Run one model server per heavyweight adapter, then compare them from the robot-side",
        "runtime through `runtime=remote`. This file is a deployment plan, not a claim that",
        "all listed external checkpoints have been locally verified.",
        "",
        "## Servers",
        "",
        "| Model | Endpoint | Install | Command | Notes |",
        "|---|---|---|---|---|",
    ]
    for entry in plan.entries:
        lines.append(
            f"| `{entry.model}` | `{entry.public_url}` | `{entry.install_hint}` | "
            f"`{entry.shell_command()}` | {entry.notes} |"
        )
    lines.extend(
        [
            "",
            "## Robot-Side Comparison",
            "",
            "```bash",
            plan.compare_command,
            "```",
            "",
            "Remote map:",
            "",
            "```text",
            plan.remote_map,
            "```",
            "",
        ]
    )
    return "\n".join(lines)
