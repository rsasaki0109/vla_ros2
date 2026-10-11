"""Asynchronous action-chunk execution: keep acting while the next chunk is inferred.

Chunked policies (SmolVLA predicts 50 steps per call) are slow per call but cover
many control ticks. Executing one buffered action per tick and requesting the
next chunk before the buffer runs dry keeps the robot moving through inference.
A chunk inferred from an observation taken ``k`` ticks ago starts ``k`` steps in,
because those steps are already in the past (latency compensation). Continuous
output therefore needs chunks of at least twice the inference latency in ticks:
the first half goes stale in flight, the second half covers the next inference.
"""

from __future__ import annotations

import math
from collections import deque

from vla_ros2.core.types import VLAAction, VLAActionChunk


def refill_threshold(
    *, avg_latency_ms: float | None, control_hz: float, margin_steps: int = 1
) -> int:
    """Buffered steps at which to request the next chunk so it lands before starvation."""

    if avg_latency_ms is None or control_hz <= 0:
        return max(1, margin_steps)
    latency_steps = math.ceil(avg_latency_ms / 1000.0 * control_hz)
    return max(1, latency_steps + margin_steps)


class ActionChunkBuffer:
    """Tick-indexed buffer: one action per control tick, newest chunk wins."""

    def __init__(self) -> None:
        self._actions: deque[VLAAction] = deque()
        self.stale_steps_dropped = 0
        self.starved_ticks = 0
        self.chunks_received = 0

    @property
    def remaining(self) -> int:
        return len(self._actions)

    def needs_refill(self, threshold: int) -> bool:
        return self.remaining <= threshold

    def push(
        self,
        prediction: VLAAction | VLAActionChunk,
        *,
        observed_tick: int,
        current_tick: int,
    ) -> int:
        """Replace the buffer with ``prediction``, skipping steps already elapsed.

        Returns how many leading steps were dropped as stale.
        """

        actions = (
            list(prediction.actions) if isinstance(prediction, VLAActionChunk) else [prediction]
        )
        elapsed = max(0, current_tick - observed_tick)
        stale = min(elapsed, len(actions))
        self._actions = deque(actions[stale:])
        self.stale_steps_dropped += stale
        self.chunks_received += 1
        return stale

    def pop(self) -> VLAAction | None:
        """Next action for this tick, or ``None`` (counted as a starved tick)."""

        if not self._actions:
            self.starved_ticks += 1
            return None
        return self._actions.popleft()

    def clear(self) -> None:
        self._actions.clear()
