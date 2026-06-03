"""Record a real adapter's action-chunk stream into a Real-Time Chunking trace.

The ``rtc-sim`` simulation runs on synthetic chunks. To take the freeze-vs-naive
continuity claim from *simulation* to a *real model runtime path*, we need the two real
inputs that fully determine an asynchronous chunk executor's output:

1. the sequence of action chunks a real policy actually predicts, and
2. the wall-clock inference latency of each prediction.

Given those, the emitted control stream of a two-thread async executor (a control thread
draining the current chunk while a background thread computes the next) is *deterministic*
-- it is exactly what :func:`simulate_emission_variable` computes from the recorded chunks
and per-cycle delays. So instead of a flaky real-time thread race we record the trace once
and replay both strategies from it reproducibly. The continuity numbers are then a real
runtime property of the recorded chunks, not synthetic; still no policy-quality claim.

``record_rtc_trace`` drives any adapter whose ``predict`` returns a ``VLAActionChunk``
(e.g. SmolVLA with ``return_action_chunk=True``). The heavy model path is gated by the
caller; this module itself imports no heavy dependency and is exercised with a fake
chunk adapter on CPU.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from vla_zoo.core.types import VLAActionChunk, VLAObservation

#: Schema identifier for the recorded trace artifact.
RTC_TRACE_SCHEMA_VERSION = "vla-zoo-rtc-trace/v1"

RTC_TRACE_NOTE = (
    "Recorded action-chunk stream + measured per-cycle inference latency from a real "
    "adapter. Replaying it through the freeze/naive scheduler reproduces what a two-thread "
    "async executor would emit. Runtime path only; no policy-quality or task-success claim."
)


@dataclass(frozen=True)
class RTCChunkEvent:
    """One recorded re-plan: the full predicted chunk and how long the inference took."""

    cycle: int
    chunk: tuple[tuple[float, ...], ...]
    inference_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "chunk": [list(action) for action in self.chunk],
            "inference_latency_ms": self.inference_latency_ms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RTCChunkEvent:
        return cls(
            cycle=int(payload["cycle"]),
            chunk=tuple(tuple(float(v) for v in row) for row in payload["chunk"]),
            inference_latency_ms=float(payload["inference_latency_ms"]),
        )


@dataclass(frozen=True)
class RTCTrace:
    """A recorded chunk stream from one model run, ready to replay through the scheduler."""

    model: str
    control_hz: float
    execute_horizon: int
    horizon: int
    events: tuple[RTCChunkEvent, ...]
    source: str
    note: str = RTC_TRACE_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RTC_TRACE_SCHEMA_VERSION,
            "model": self.model,
            "control_hz": self.control_hz,
            "execute_horizon": self.execute_horizon,
            "horizon": self.horizon,
            "source": self.source,
            "events": [event.to_dict() for event in self.events],
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RTCTrace:
        return cls(
            model=str(payload["model"]),
            control_hz=float(payload["control_hz"]),
            execute_horizon=int(payload["execute_horizon"]),
            horizon=int(payload["horizon"]),
            events=tuple(RTCChunkEvent.from_dict(e) for e in payload["events"]),
            source=str(payload["source"]),
            note=str(payload.get("note", RTC_TRACE_NOTE)),
        )


def _chunk_to_rows(prediction: VLAActionChunk | Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(prediction, VLAActionChunk):
        msg = "adapter must return a VLAActionChunk (set return_action_chunk=True)"
        raise TypeError(msg)
    return tuple(tuple(float(v) for v in action.tolist()) for action in prediction.actions)


def record_rtc_trace(
    adapter: Any,
    observations: Sequence[VLAObservation],
    *,
    control_hz: float = 30.0,
    execute_horizon: int = 8,
    model_name: str | None = None,
    source: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RTCTrace:
    """Drive ``adapter`` over ``observations``, recording each chunk + its inference latency.

    The adapter's ``predict`` must return a :class:`VLAActionChunk`. Every chunk must share
    the same horizon. ``clock`` is injectable so tests can supply deterministic timings;
    real runs use ``time.perf_counter``.
    """

    if not observations:
        msg = "at least one observation is required"
        raise ValueError(msg)
    if control_hz <= 0:
        msg = "control_hz must be positive"
        raise ValueError(msg)

    pairs: list[tuple[tuple[tuple[float, ...], ...], float]] = []
    for observation in observations:
        start = clock()
        prediction = adapter.predict(observation=observation)
        latency_ms = (clock() - start) * 1000.0
        pairs.append((_chunk_to_rows(prediction), latency_ms))

    resolved_model = model_name or str(getattr(adapter, "name", "unknown"))
    return build_trace_from_chunks(
        pairs,
        model=resolved_model,
        control_hz=control_hz,
        execute_horizon=execute_horizon,
        source=source or "rtc-executor-recording",
    )


def build_trace_from_chunks(
    chunk_latency_pairs: Sequence[tuple[Sequence[Sequence[float]], float]],
    *,
    model: str,
    control_hz: float,
    execute_horizon: int,
    source: str,
) -> RTCTrace:
    """Assemble an :class:`RTCTrace` from recorded (chunk rows, latency ms) pairs.

    Validates that every chunk shares one horizon and that ``execute_horizon`` fits it.
    Used both by :func:`record_rtc_trace` and by callers that capture chunks through a
    rollout sink (e.g. the PyBullet action probe).
    """

    if not chunk_latency_pairs:
        msg = "at least one chunk is required"
        raise ValueError(msg)
    if control_hz <= 0:
        msg = "control_hz must be positive"
        raise ValueError(msg)

    events: list[RTCChunkEvent] = []
    horizon: int | None = None
    for cycle, (rows, latency_ms) in enumerate(chunk_latency_pairs):
        chunk = tuple(tuple(float(v) for v in row) for row in rows)
        if horizon is None:
            horizon = len(chunk)
        elif len(chunk) != horizon:
            msg = f"chunk {cycle} has {len(chunk)} actions, expected {horizon}"
            raise ValueError(msg)
        events.append(
            RTCChunkEvent(cycle=cycle, chunk=chunk, inference_latency_ms=float(latency_ms))
        )

    assert horizon is not None  # guarded by the empty-pairs check
    if not 1 <= execute_horizon <= horizon:
        msg = f"execute_horizon must be in [1, {horizon}] for this chunk horizon"
        raise ValueError(msg)

    return RTCTrace(
        model=model,
        control_hz=control_hz,
        execute_horizon=execute_horizon,
        horizon=horizon,
        events=tuple(events),
        source=source,
    )


def trace_delays_ticks(trace: RTCTrace) -> list[int]:
    """Convert each event's measured latency into integer control-tick delays."""

    return [round(event.inference_latency_ms / 1000.0 * trace.control_hz) for event in trace.events]


def compare_trace_strategies(trace: RTCTrace) -> Any:
    """Replay a recorded trace through naive-async and RTC-freeze, returning an RTCSimReport.

    Uses the real per-cycle inference delays (latency -> control ticks), so the continuity
    numbers are a real runtime property of the recorded chunk stream.
    """

    import numpy as np

    from vla_zoo.runtime.realtime_chunking import (
        RealtimeChunkingConfig,
        RTCSimReport,
        simulate_emission_variable,
    )

    if not trace.events:
        msg = "trace has no events"
        raise ValueError(msg)
    chunks = [np.asarray(event.chunk, dtype=np.float32) for event in trace.events]
    delays = trace_delays_ticks(trace)
    naive, late = simulate_emission_variable(
        chunks, delays, horizon=trace.horizon, execute_horizon=trace.execute_horizon, freeze=False
    )
    rtc, _ = simulate_emission_variable(
        chunks, delays, horizon=trace.horizon, execute_horizon=trace.execute_horizon, freeze=True
    )
    if naive.mean_boundary_jump > 0:
        reduction = (naive.mean_boundary_jump - rtc.mean_boundary_jump) / naive.mean_boundary_jump
    else:
        reduction = 0.0

    # A representative delay for display only; the emission used the real per-cycle values.
    p50 = int(np.median(delays[1:])) if len(delays) > 1 else 0
    display_delay = max(0, min(p50, trace.horizon - trace.execute_horizon))
    config = RealtimeChunkingConfig(
        horizon=trace.horizon,
        execute_horizon=trace.execute_horizon,
        inference_delay_ticks=display_delay,
        control_hz=trace.control_hz,
    )
    source = (
        f"{trace.source} (model {trace.model}, real per-cycle delays, "
        f"p50 {p50} ticks @ {trace.control_hz:g} Hz, late cycles {late})"
    )
    return RTCSimReport(
        config=config,
        chunk_count=len(chunks),
        action_dim=int(chunks[0].shape[1]),
        naive=naive,
        rtc=rtc,
        boundary_jump_reduction=float(reduction),
        source=source,
    )
