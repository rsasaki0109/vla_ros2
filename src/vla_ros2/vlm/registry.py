"""Built-in VLM table. Kept separate from the VLA adapter registry on purpose:
VLMs answer questions and never produce robot actions."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from vla_ros2.core.errors import UnknownModelError
from vla_ros2.vlm.base import BaseVLM


@dataclass(frozen=True)
class VLMInfo:
    name: str
    target: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    install_hint: str | None = None

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


_BUILTINS: tuple[VLMInfo, ...] = (
    VLMInfo(
        name="dummy",
        target="vla_ros2.vlm.dummy:DummyVLM",
        description="Fixed-answer VLM for tests and ROS2 wiring checks.",
    ),
    VLMInfo(
        name="smolvlm",
        target="vla_ros2.vlm.smolvlm:SmolVLMAdapter",
        aliases=("smolvlm2",),
        description="Hugging Face SmolVLM2 (default 500M Video-Instruct).",
        install_hint='pip install "vla_ros2[vlm]"',
    ),
)


def list_vlms() -> list[VLMInfo]:
    return list(_BUILTINS)


def get_vlm_info(name: str) -> VLMInfo:
    key = name.strip().lower()
    for info in _BUILTINS:
        if key in info.all_names:
            return info
    known = ", ".join(info.name for info in _BUILTINS)
    msg = f"Unknown VLM {name!r}. Known VLMs: {known}"
    raise UnknownModelError(msg)


def load_vlm(name: str, **kwargs: Any) -> BaseVLM:
    """Instantiate a registered VLM for local inference."""

    info = get_vlm_info(name)
    module_name, attr = info.target.split(":", 1)
    factory = getattr(import_module(module_name), attr)
    vlm = factory(**kwargs)
    if not isinstance(vlm, BaseVLM):
        msg = f"{info.target} did not return a BaseVLM"
        raise TypeError(msg)
    return vlm
