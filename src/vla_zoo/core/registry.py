from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

from vla_zoo.core.errors import ConfigurationError, MissingDependencyError, UnknownModelError
from vla_zoo.core.model import BaseVLA

ENTRY_POINT_GROUP = "vla_zoo.adapters"


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    target: str
    source: str = "built-in"
    aliases: tuple[str, ...] = ()
    experimental: bool = False
    domain: str | None = None
    description: str = ""
    install_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


ModelAdapterFactory = Callable[..., BaseVLA]


_BUILTINS: dict[str, AdapterInfo] = {
    "dummy": AdapterInfo(
        name="dummy",
        target="vla_zoo.adapters.dummy:DummyAdapter",
        description="Always-available zero-action adapter for tests and dry runs.",
        metadata={
            "family": "dry-run baseline",
            "compare_role": "CI/runtime smoke sanity check",
            "input_requirements": (
                "image optional",
                "instruction optional",
                "state optional",
            ),
            "output": "zero 7-DoF end-effector delta",
            "action_space": "eef_delta",
            "action_shape": "(7,)",
            "control_hz": "5",
            "action_chunks": "optional via chunk_size",
            "proprioception": "not required",
            "local_runtime": "supported",
            "remote_runtime": "supported",
            "dependency_profile": "base install",
            "license_caveat": "none",
        },
    ),
    "random": AdapterInfo(
        name="random",
        target="vla_zoo.adapters.baselines:RandomAdapter",
        aliases=("random-baseline",),
        description="Always-available seeded random-action baseline.",
        metadata={
            "baseline": True,
            "family": "stochastic baseline",
            "compare_role": "action plumbing and visualization stress check",
            "input_requirements": (
                "instruction optional",
                "image optional",
                "state optional",
            ),
            "output": "seeded random 7-DoF end-effector delta",
            "action_space": "eef_delta",
            "action_shape": "(7,)",
            "control_hz": "5",
            "action_chunks": "no",
            "proprioception": "not required",
            "local_runtime": "supported",
            "remote_runtime": "supported",
            "dependency_profile": "base install",
            "license_caveat": "none",
        },
    ),
    "scripted": AdapterInfo(
        name="scripted",
        target="vla_zoo.adapters.baselines:ScriptedAdapter",
        aliases=("heuristic", "rule-based"),
        description="Always-available phase-aware scripted baseline.",
        metadata={
            "baseline": True,
            "family": "rule-based baseline",
            "compare_role": "upper-bound sanity check for the scripted smoke scene",
            "input_requirements": (
                "phase metadata",
                "instruction optional",
                "image optional",
            ),
            "output": "phase-aware 7-DoF end-effector delta",
            "action_space": "eef_delta",
            "action_shape": "(7,)",
            "control_hz": "5",
            "action_chunks": "no",
            "proprioception": "not required",
            "local_runtime": "supported",
            "remote_runtime": "supported",
            "dependency_profile": "base install",
            "license_caveat": "none",
        },
    ),
    "openvla": AdapterInfo(
        name="openvla",
        target="vla_zoo.adapters.openvla:OpenVLAAdapter",
        description="OpenVLA Hugging Face adapter.",
        install_hint='pip install "vla_zoo[openvla]"',
        metadata={
            "family": "VLA foundation model",
            "compare_role": "single-image VLA reference adapter",
            "input_requirements": (
                "single RGB image",
                "natural language instruction",
                "optional unnormalization key",
            ),
            "output": "OpenVLA-style 7-DoF action",
            "action_space": "eef_delta",
            "action_shape": "(7,)",
            "control_hz": "model/robot dependent",
            "action_chunks": "no",
            "proprioception": "not required by default adapter",
            "local_runtime": "supported with optional ML dependencies",
            "remote_runtime": "recommended for robot-side ROS2",
            "dependency_profile": "torch, transformers, HF weights",
            "license_caveat": "external project and model license apply",
        },
    ),
    "pi0": AdapterInfo(
        name="pi0",
        target="vla_zoo.adapters.pi0:Pi0Adapter",
        aliases=("openpi", "pi0-fast", "pi05"),
        experimental=True,
        description="Remote-first pi0/openpi adapter with optional local LeRobot loading.",
        install_hint='pip install "vla_zoo[openpi]"',
        metadata={
            "family": "pi-family VLA",
            "compare_role": "remote-first action-chunk VLA target",
            "input_requirements": (
                "images per policy config",
                "natural language instruction",
                "robot state expected",
            ),
            "output": "policy-specific continuous manipulation action",
            "action_space": "custom",
            "action_shape": "checkpoint-specific; lerobot/pi0 is (6,), lerobot/pi0_base is (32,)",
            "control_hz": "policy/server dependent",
            "action_chunks": "expected",
            "proprioception": "expected",
            "local_runtime": "disabled by default; enable_local=True in a dedicated GPU env",
            "remote_runtime": "recommended",
            "dependency_profile": "LeRobot/openpi stack in serving environment",
            "license_caveat": "external project and checkpoint license apply",
        },
    ),
    "smolvla": AdapterInfo(
        name="smolvla",
        target="vla_zoo.adapters.smolvla:SmolVLAAdapter",
        aliases=("lerobot-smolvla",),
        experimental=True,
        description="LeRobot SmolVLA adapter for compact local VLA inference.",
        install_hint='pip install "vla_zoo[smolvla]"',
        metadata={
            "family": "LeRobot policy",
            "compare_role": "multi-camera/state/action-chunk compact VLA target",
            "input_requirements": (
                "multi-camera images",
                "natural language instruction",
                "robot state",
            ),
            "output": "policy-specific continuous action",
            "action_space": "custom",
            "action_shape": "checkpoint-specific; lerobot/smolvla_base is (6,)",
            "control_hz": "policy/robot dependent",
            "action_chunks": "internal queue; chunk output optional",
            "proprioception": "required by typical deployments",
            "local_runtime": "supported with optional LeRobot dependencies",
            "remote_runtime": "recommended",
            "dependency_profile": "lerobot[smolvla], torch, HF weights",
            "license_caveat": "external project, dataset, and checkpoint licenses apply",
        },
    ),
    "groot": AdapterInfo(
        name="groot",
        target="vla_zoo.adapters.groot:GR00TAdapter",
        aliases=("gr00t", "isaac-groot"),
        experimental=True,
        domain="humanoid/generalist",
        description="Experimental placeholder for Isaac GR00T-style adapters.",
        install_hint="Install Isaac GR00T dependencies in the serving environment.",
        metadata={
            "family": "humanoid/generalist foundation model",
            "compare_role": "experimental humanoid/generalist adapter target",
            "input_requirements": (
                "multimodal observations",
                "instruction/task context",
                "robot state expected",
            ),
            "output": "humanoid/generalist action interface",
            "action_space": "custom",
            "action_shape": "adapter-specific",
            "control_hz": "stack dependent",
            "action_chunks": "adapter-specific",
            "proprioception": "expected",
            "local_runtime": "experimental placeholder",
            "remote_runtime": "recommended",
            "dependency_profile": "Isaac GR00T stack in serving environment",
            "license_caveat": "external NVIDIA project and model license apply",
        },
    ),
}

_ENTRY_POINT_CACHE: dict[str, AdapterInfo] | None = None


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _load_target(target: str) -> type[BaseVLA]:
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        msg = f"Invalid adapter target {target!r}"
        raise ConfigurationError(msg)
    try:
        module = __import__(module_name, fromlist=[attr])
    except MissingDependencyError:
        raise
    except ImportError as exc:
        raise MissingDependencyError(str(exc)) from exc
    adapter_cls = getattr(module, attr)
    if not isinstance(adapter_cls, type) or not issubclass(adapter_cls, BaseVLA):
        msg = f"Entry point {target!r} did not resolve to a BaseVLA subclass"
        raise ConfigurationError(msg)
    return adapter_cls


def _entry_point_infos() -> dict[str, AdapterInfo]:
    global _ENTRY_POINT_CACHE
    if _ENTRY_POINT_CACHE is not None:
        return _ENTRY_POINT_CACHE

    infos: dict[str, AdapterInfo] = {}
    try:
        entry_points = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:
        entry_points = metadata.entry_points().select(group=ENTRY_POINT_GROUP)
    for entry_point in entry_points:
        normalized = _normalize_name(entry_point.name)
        if normalized in _BUILTINS:
            continue
        infos[normalized] = AdapterInfo(
            name=normalized,
            target=f"{entry_point.module}:{entry_point.attr}",
            source="entry-point",
            description=f"External adapter registered by {entry_point.value}.",
        )
    _ENTRY_POINT_CACHE = infos
    return infos


def _adapter_index() -> dict[str, AdapterInfo]:
    index: dict[str, AdapterInfo] = {}
    for info in (*_BUILTINS.values(), *_entry_point_infos().values()):
        for alias in info.all_names:
            index[_normalize_name(alias)] = info
    return index


def get_adapter_info(name: str) -> AdapterInfo:
    """Return metadata for a registered adapter or alias."""

    normalized = _normalize_name(name)
    try:
        return _adapter_index()[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(info.name for info in list_models()))
        msg = f"Unknown VLA model {name!r}. Available models: {choices}"
        raise UnknownModelError(msg) from exc


def list_models() -> list[AdapterInfo]:
    """List canonical adapters known to the built-in and entry-point registries."""

    infos = [*_BUILTINS.values(), *_entry_point_infos().values()]
    return sorted(infos, key=lambda item: item.name)


def load_model(
    name: str,
    *,
    runtime: str = "local",
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BaseVLA:
    """Load a VLA model adapter by name."""

    merged_config = dict(config or {})
    merged_config.update(kwargs)

    if runtime == "remote":
        from vla_zoo.runtime.remote import RemoteVLAClient

        remote_url = str(merged_config.pop("remote_url", "http://localhost:8000"))
        return RemoteVLAClient(model_name=name, remote_url=remote_url, **merged_config)
    if runtime != "local":
        msg = f"Unsupported runtime {runtime!r}; expected 'local' or 'remote'"
        raise ConfigurationError(msg)

    info = get_adapter_info(name)
    try:
        adapter_cls = _load_target(info.target)
        return adapter_cls.from_config(**merged_config)
    except MissingDependencyError as exc:
        hint = info.install_hint
        if hint and hint not in str(exc):
            msg = f"{exc}. Install optional dependencies with: {hint}"
            raise MissingDependencyError(msg) from exc
        raise
