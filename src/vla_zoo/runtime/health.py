from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeHealth:
    ready: bool
    model_name: str
    dry_run: bool
    last_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    action_rate_hz: float = 0.0
    dropped_frames: int = 0
    status_text: str = "ok"
