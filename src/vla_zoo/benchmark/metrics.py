from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass
class MetricsAccumulator:
    latencies_ms: list[float] = field(default_factory=list)
    successes: int = 0
    episodes: int = 0
    exceptions: int = 0

    def summary(self) -> dict[str, float | int]:
        sorted_latencies = sorted(self.latencies_ms)
        p50 = median(sorted_latencies) if sorted_latencies else 0.0
        p95_index = int(0.95 * (len(sorted_latencies) - 1)) if sorted_latencies else 0
        p95 = sorted_latencies[p95_index] if sorted_latencies else 0.0
        return {
            "success_rate": self.successes / self.episodes if self.episodes else 0.0,
            "action_latency_ms_p50": float(p50),
            "action_latency_ms_p95": float(p95),
            "exception_count": self.exceptions,
            "timeout_count": 0,
            "dropped_frame_count": 0,
            "action_clip_rate": 0.0,
        }
