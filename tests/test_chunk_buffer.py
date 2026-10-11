from __future__ import annotations

import numpy as np

from vla_ros2.core.types import ActionSpec, VLAAction, VLAActionChunk
from vla_ros2.runtime.chunking import ActionChunkBuffer, refill_threshold

SPEC = ActionSpec(action_space="custom", shape=(1,))


def _chunk(start: int, size: int) -> VLAActionChunk:
    return VLAActionChunk(
        actions=[
            VLAAction(data=np.asarray([float(start + i)], dtype=np.float32), spec=SPEC)
            for i in range(size)
        ]
    )


def _value(action: VLAAction | None) -> float | None:
    return None if action is None else float(action.data[0])


def test_refill_threshold_covers_latency() -> None:
    assert refill_threshold(avg_latency_ms=None, control_hz=10.0) == 1
    # 1 s inference at 10 Hz = 10 ticks, plus one tick of margin.
    assert refill_threshold(avg_latency_ms=1000.0, control_hz=10.0) == 11
    assert refill_threshold(avg_latency_ms=10.0, control_hz=10.0, margin_steps=0) == 1


def test_push_skips_steps_elapsed_since_observation() -> None:
    buffer = ActionChunkBuffer()
    dropped = buffer.push(_chunk(0, 5), observed_tick=10, current_tick=13)
    assert dropped == 3
    assert buffer.remaining == 2
    assert _value(buffer.pop()) == 3.0
    assert _value(buffer.pop()) == 4.0
    assert buffer.pop() is None
    assert buffer.starved_ticks == 1
    assert buffer.stale_steps_dropped == 3


def test_newer_chunk_replaces_remaining_actions() -> None:
    buffer = ActionChunkBuffer()
    buffer.push(_chunk(0, 5), observed_tick=0, current_tick=0)
    buffer.pop()
    buffer.push(_chunk(100, 5), observed_tick=1, current_tick=1)
    assert _value(buffer.pop()) == 100.0
    assert buffer.remaining == 4
    assert buffer.chunks_received == 2


def test_fully_stale_chunk_leaves_buffer_empty_and_single_action_works() -> None:
    buffer = ActionChunkBuffer()
    assert buffer.push(_chunk(0, 2), observed_tick=0, current_tick=5) == 2
    assert buffer.remaining == 0
    single = VLAAction(data=np.zeros(1, dtype=np.float32), spec=SPEC)
    buffer.push(single, observed_tick=5, current_tick=5)
    assert buffer.remaining == 1
    buffer.clear()
    assert buffer.remaining == 0


def _simulate(*, async_mode: bool, ticks: int, latency_ticks: int, chunk: int) -> int:
    """Return how many ticks published no action for a policy with fixed latency."""

    buffer = ActionChunkBuffer()
    pending_done_at: int | None = None
    pending_observed = 0
    idle = 0
    for tick in range(ticks):
        if pending_done_at is not None and tick >= pending_done_at:
            buffer.push(_chunk(0, chunk), observed_tick=pending_observed, current_tick=tick)
            pending_done_at = None
        threshold = latency_ticks + 1 if async_mode else 0
        if pending_done_at is None and buffer.needs_refill(threshold):
            pending_done_at, pending_observed = tick + latency_ticks, tick
        if buffer.pop() is None:
            idle += 1
    return idle


def test_async_execution_removes_inference_stalls() -> None:
    # 50-step chunks, 5-tick inference, 200 ticks: refilling only when empty stalls
    # for the whole inference every chunk; refilling ahead stalls only at startup.
    sync_idle = _simulate(async_mode=False, ticks=200, latency_ticks=5, chunk=50)
    async_idle = _simulate(async_mode=True, ticks=200, latency_ticks=5, chunk=50)
    assert async_idle == 5  # first chunk only
    assert sync_idle > 3 * async_idle
