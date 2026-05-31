from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from vla_zoo.core.model import BaseVLA
from vla_zoo.core.types import VLAAction, VLAActionChunk, VLAObservation


@dataclass
class SchedulerState:
    queued_actions: int
    requests: int
    consumed_actions: int


class ActionChunkScheduler:
    """Small queue-based scheduler for action chunks."""

    def __init__(self, *, refill_threshold: int = 1, max_queue_size: int = 8) -> None:
        if refill_threshold < 0:
            msg = "refill_threshold must be non-negative"
            raise ValueError(msg)
        if max_queue_size <= 0:
            msg = "max_queue_size must be positive"
            raise ValueError(msg)
        self.refill_threshold = refill_threshold
        self.max_queue_size = max_queue_size
        self._queue: deque[VLAAction] = deque()
        self._requests = 0
        self._consumed = 0

    @property
    def state(self) -> SchedulerState:
        return SchedulerState(
            queued_actions=len(self._queue),
            requests=self._requests,
            consumed_actions=self._consumed,
        )

    def push(self, prediction: VLAAction | VLAActionChunk) -> None:
        actions = prediction.actions if isinstance(prediction, VLAActionChunk) else [prediction]
        for action in actions:
            if len(self._queue) >= self.max_queue_size:
                break
            self._queue.append(action)

    def pop(self) -> VLAAction | None:
        if not self._queue:
            return None
        self._consumed += 1
        return self._queue.popleft()

    def needs_refill(self) -> bool:
        return len(self._queue) <= self.refill_threshold

    def refill(self, model: BaseVLA, observation: VLAObservation) -> None:
        if not self.needs_refill():
            return
        self._requests += 1
        self.push(model.predict(observation=observation))
