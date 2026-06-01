"""Configuration dataclasses shared by runtime entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class RuntimeConfig:
    model_name: str = "dummy"
    runtime: Literal["local", "remote"] = "local"
    dry_run: bool = True
    remote_url: str = "http://localhost:8000"
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)
