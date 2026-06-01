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
    ),
    "openvla": AdapterInfo(
        name="openvla",
        target="vla_zoo.adapters.openvla:OpenVLAAdapter",
        description="OpenVLA Hugging Face adapter.",
        install_hint='pip install "vla_zoo[openvla]"',
    ),
    "pi0": AdapterInfo(
        name="pi0",
        target="vla_zoo.adapters.pi0:Pi0Adapter",
        aliases=("openpi", "pi0-fast", "pi05"),
        experimental=True,
        description="Remote-first placeholder for openpi/pi0 model family.",
        install_hint="Install openpi dependencies in the serving environment.",
    ),
    "smolvla": AdapterInfo(
        name="smolvla",
        target="vla_zoo.adapters.smolvla:SmolVLAAdapter",
        aliases=("lerobot-smolvla",),
        experimental=True,
        description="Placeholder for LeRobot SmolVLA policies.",
        install_hint="Install LeRobot dependencies in the serving environment.",
    ),
    "groot": AdapterInfo(
        name="groot",
        target="vla_zoo.adapters.groot:GR00TAdapter",
        aliases=("gr00t", "isaac-groot"),
        experimental=True,
        domain="humanoid/generalist",
        description="Experimental placeholder for Isaac GR00T-style adapters.",
        install_hint="Install Isaac GR00T dependencies in the serving environment.",
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
