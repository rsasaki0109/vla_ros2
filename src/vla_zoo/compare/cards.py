from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from vla_zoo.compare.profiles import MethodProfile, profile_from_adapter_info
from vla_zoo.core.registry import AdapterInfo, get_adapter_info

DEFAULT_CARD_MODELS = ("dummy", "scripted", "random", "openvla", "pi0", "smolvla", "groot")


def _table_value(value: object) -> str:
    text = str(value) if value not in (None, "") else "-"
    return text.replace("\n", " ").replace("|", "\\|")


def _csv(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "-"


def _input_bullets(values: Sequence[str]) -> str:
    if not values:
        return "- declared by adapter"
    return "\n".join(f"- {value}" for value in values)


def _command_lines(profile: MethodProfile) -> list[str]:
    lines = [f"vla-zoo info {profile.name}"]
    if profile.dependency_profile == "base install":
        lines.append(
            f'vla-zoo predict --model {profile.name} --instruction "pick up the red block"'
        )
    else:
        lines.append(f"vla-zoo serve-plan --models {profile.name}")
    return lines


def adapter_card_payload(
    info: AdapterInfo,
    *,
    status: str = "unknown",
) -> dict[str, Any]:
    """Return a machine-readable adapter card payload without importing adapter code."""

    profile = profile_from_adapter_info(info, status=status)
    metadata = dict(info.metadata)
    return {
        "name": info.name,
        "source": info.source,
        "target": info.target,
        "aliases": info.aliases,
        "experimental": info.experimental,
        "domain": info.domain,
        "description": info.description,
        "install_hint": info.install_hint,
        "status": status,
        "runtime_contract": {
            "family": profile.family,
            "compare_role": profile.compare_role,
            "input_requirements": profile.input_requirements,
            "output": profile.output,
            "action_space": profile.action_space,
            "action_shape": profile.action_shape,
            "control_hz": profile.control_hz,
            "action_chunks": profile.action_chunks,
            "proprioception": profile.proprioception,
            "local_runtime": profile.local_runtime,
            "remote_runtime": profile.remote_runtime,
            "dependency_profile": profile.dependency_profile,
            "license_caveat": profile.license_caveat,
        },
        "verification": metadata.get(
            "verification",
            "No adapter-specific verification note has been declared yet.",
        ),
        "metadata": metadata,
    }


def format_adapter_card_markdown(
    info: AdapterInfo,
    *,
    status: str = "unknown",
) -> str:
    """Format one adapter capability card as Markdown."""

    profile = profile_from_adapter_info(info, status=status)
    metadata = info.metadata
    verification = str(
        metadata.get("verification", "No adapter-specific verification note has been declared yet.")
    )
    upstream = str(metadata.get("upstream_project", "-"))
    default_checkpoint = str(metadata.get("default_checkpoint", "-"))
    caveat = profile.license_caveat

    lines = [
        f"# {info.name} Adapter Card",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Source | {_table_value(info.source)} |",
        f"| Target | `{_table_value(info.target)}` |",
        f"| Upstream project | {_table_value(upstream)} |",
        f"| Default checkpoint | `{_table_value(default_checkpoint)}` |",
        f"| Aliases | {_table_value(_csv(info.aliases))} |",
        f"| Status | {_table_value(status)} |",
        f"| Experimental | {_table_value(str(info.experimental).lower())} |",
        f"| Domain | {_table_value(info.domain)} |",
        f"| Install hint | `{_table_value(info.install_hint)}` |",
        "",
        "## Runtime Contract",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Family | {_table_value(profile.family)} |",
        f"| Role | {_table_value(profile.compare_role)} |",
        f"| Action space | `{_table_value(profile.action_space)}` |",
        f"| Action shape | `{_table_value(profile.action_shape)}` |",
        f"| Output | {_table_value(profile.output)} |",
        f"| Control Hz | {_table_value(profile.control_hz)} |",
        f"| Action chunks | {_table_value(profile.action_chunks)} |",
        f"| Proprioception | {_table_value(profile.proprioception)} |",
        f"| Local runtime | {_table_value(profile.local_runtime)} |",
        f"| Remote runtime | {_table_value(profile.remote_runtime)} |",
        f"| Dependencies | {_table_value(profile.dependency_profile)} |",
        f"| License caveat | {_table_value(caveat)} |",
        "",
        "## Inputs",
        "",
        _input_bullets(profile.input_requirements),
        "",
        "## Verification Status",
        "",
        verification,
        "",
        "## Operational Caveats",
        "",
        "- This card describes the vla_zoo runtime contract, not task success on a robot.",
        (
            "- External model weights, datasets, and upstream licenses are not redistributed "
            "by vla_zoo."
        ),
        (
            "- Real robot deployment still needs calibrated cameras, state mapping, action "
            "clipping, watchdogs, and a hardware-specific bridge."
        ),
        "",
        "## Useful Commands",
        "",
        "```bash",
        *_command_lines(profile),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _adapter_infos(models: Sequence[str] | None = None) -> list[AdapterInfo]:
    requested = models if models is not None else DEFAULT_CARD_MODELS
    infos: list[AdapterInfo] = []
    seen: set[str] = set()
    for model in requested:
        info = get_adapter_info(model)
        if info.name in seen:
            continue
        seen.add(info.name)
        infos.append(info)
    return infos


def format_adapter_cards_index(
    infos: Sequence[AdapterInfo],
    *,
    status_provider: Callable[[str], str] | None = None,
) -> str:
    """Format an index page for adapter cards."""

    lines = [
        "# vla_zoo Adapter Cards",
        "",
        "These cards document adapter runtime contracts. They do not claim model quality, "
        "zero-shot success, or hardware readiness.",
        "",
        "| Adapter | Family | Action | Runtime | Verification | Card |",
        "|---|---|---|---|---|---|",
    ]
    for info in infos:
        status = status_provider(info.name) if status_provider is not None else "unknown"
        profile = profile_from_adapter_info(info, status=status)
        verification = str(info.metadata.get("verification", "-"))
        action = f"{profile.action_space} {profile.action_shape}"
        runtime = f"local: {profile.local_runtime}; remote: {profile.remote_runtime}"
        lines.append(
            f"| `{info.name}` | {_table_value(profile.family)} | {_table_value(action)} | "
            f"{_table_value(runtime)} | {_table_value(verification)} | "
            f"[card]({info.name}.md) |"
        )
    lines.extend(
        [
            "",
            "Generate these files from the registry:",
            "",
            "```bash",
            "vla-zoo compare cards --out-dir docs/adapters",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def write_adapter_cards(
    out_dir: Path,
    *,
    models: Sequence[str] | None = None,
    status_provider: Callable[[str], str] | None = None,
) -> list[Path]:
    """Write adapter card Markdown files and return the generated paths."""

    infos = _adapter_infos(models)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    index_path = out_dir / "README.md"
    index_path.write_text(
        format_adapter_cards_index(infos, status_provider=status_provider),
        encoding="utf-8",
    )
    paths.append(index_path)

    for info in infos:
        status = status_provider(info.name) if status_provider is not None else "unknown"
        path = out_dir / f"{info.name}.md"
        path.write_text(format_adapter_card_markdown(info, status=status), encoding="utf-8")
        paths.append(path)
    return paths
