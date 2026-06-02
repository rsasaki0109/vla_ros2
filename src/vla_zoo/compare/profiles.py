from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

from vla_zoo.core.registry import AdapterInfo, list_models


@dataclass(frozen=True)
class MethodProfile:
    """Human-readable comparison profile for a VLA adapter or baseline method."""

    name: str
    source: str
    status: str
    family: str
    compare_role: str
    input_requirements: tuple[str, ...]
    output: str
    action_space: str
    action_shape: str
    control_hz: str
    action_chunks: str
    proprioception: str
    local_runtime: str
    remote_runtime: str
    dependency_profile: str
    license_caveat: str
    aliases: tuple[str, ...] = ()
    experimental: bool = False
    domain: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metadata_string(metadata: dict[str, Any], key: str, default: str) -> str:
    value = metadata.get(key, default)
    if isinstance(value, str):
        return value
    return str(value)


def _metadata_tuple(
    metadata: dict[str, Any],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = metadata.get(key, default)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return default


def profile_from_adapter_info(
    info: AdapterInfo,
    *,
    status: str = "unknown",
) -> MethodProfile:
    metadata = info.metadata
    fallback_family = "external adapter" if info.source == "entry-point" else "adapter"
    return MethodProfile(
        name=info.name,
        source=info.source,
        status=status,
        family=_metadata_string(metadata, "family", fallback_family),
        compare_role=_metadata_string(metadata, "compare_role", info.description or "-"),
        input_requirements=_metadata_tuple(
            metadata,
            "input_requirements",
            ("declared by adapter",),
        ),
        output=_metadata_string(metadata, "output", "declared by adapter"),
        action_space=_metadata_string(metadata, "action_space", "custom"),
        action_shape=_metadata_string(metadata, "action_shape", "adapter-specific"),
        control_hz=_metadata_string(metadata, "control_hz", "adapter-specific"),
        action_chunks=_metadata_string(metadata, "action_chunks", "adapter-specific"),
        proprioception=_metadata_string(metadata, "proprioception", "adapter-specific"),
        local_runtime=_metadata_string(metadata, "local_runtime", "adapter-specific"),
        remote_runtime=_metadata_string(metadata, "remote_runtime", "supported through vla_zoo"),
        dependency_profile=_metadata_string(metadata, "dependency_profile", "external package"),
        license_caveat=_metadata_string(metadata, "license_caveat", "external adapter license"),
        aliases=info.aliases,
        experimental=info.experimental,
        domain=info.domain,
    )


def method_profiles(
    *,
    adapters: list[AdapterInfo] | None = None,
    status_provider: Callable[[str], str] | None = None,
) -> list[MethodProfile]:
    infos = adapters if adapters is not None else list_models()
    profiles: list[MethodProfile] = []
    for info in infos:
        status = status_provider(info.name) if status_provider is not None else "unknown"
        profiles.append(profile_from_adapter_info(info, status=status))
    return profiles


def format_method_profiles_markdown(
    profiles: list[MethodProfile],
    *,
    title: str = "VLA Method Profiles",
) -> str:
    lines = [
        f"## {title}",
        "",
        "| Method | Family | Role | Inputs | Action | Chunks | Local | Remote | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for profile in profiles:
        inputs = "<br>".join(profile.input_requirements)
        action = f"{profile.action_space} {profile.action_shape}: {profile.output}"
        lines.append(
            f"| `{profile.name}` | {profile.family} | {profile.compare_role} | "
            f"{inputs} | {action} | {profile.action_chunks} | {profile.local_runtime} | "
            f"{profile.remote_runtime} | {profile.status} |"
        )
    lines.extend(
        [
            "",
            "These profiles describe runtime integration shape, not model quality. "
            "External model weights, datasets, and licenses are not redistributed by vla_zoo.",
        ]
    )
    return "\n".join(lines) + "\n"
