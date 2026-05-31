from __future__ import annotations

from vla_zoo import load_model
from vla_zoo.core.types import VLAObservation
from vla_zoo.runtime.scheduler import ActionChunkScheduler


def test_scheduler_refills_from_dummy_action_chunk() -> None:
    model = load_model("dummy", chunk_size=3)
    scheduler = ActionChunkScheduler(refill_threshold=1, max_queue_size=4)
    observation = VLAObservation(instruction="test")

    assert scheduler.needs_refill()
    scheduler.refill(model, observation)

    assert scheduler.state.requests == 1
    assert scheduler.state.queued_actions == 3

    first = scheduler.pop()
    second = scheduler.pop()

    assert first is not None
    assert second is not None
    assert scheduler.state.consumed_actions == 2
    assert scheduler.state.queued_actions == 1
    assert scheduler.needs_refill()


def test_scheduler_does_not_refill_when_queue_is_above_threshold() -> None:
    model = load_model("dummy", chunk_size=3)
    scheduler = ActionChunkScheduler(refill_threshold=1, max_queue_size=4)
    observation = VLAObservation(instruction="test")

    scheduler.refill(model, observation)
    scheduler.refill(model, observation)

    assert scheduler.state.requests == 1
    assert scheduler.state.queued_actions == 3
