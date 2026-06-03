from __future__ import annotations

import numpy as np
import pytest

from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation
from vla_zoo.runtime.realtime_chunking import RTC_SIM_SCHEMA_VERSION, simulate_emission_variable
from vla_zoo.runtime.rtc_executor import (
    RTC_TRACE_SCHEMA_VERSION,
    RTCTrace,
    build_trace_from_chunks,
    compare_trace_strategies,
    record_rtc_trace,
    trace_delays_ticks,
)

_SPEC = ActionSpec(action_space="eef_delta", shape=(3,))


class _FakeChunkAdapter:
    """Returns a deterministic action chunk; mode offset per call creates boundary jumps."""

    name = "fake"

    def __init__(self, horizon: int, *, mode_strength: float = 0.6) -> None:
        self.horizon = horizon
        self.mode_strength = mode_strength
        self._rng = np.random.default_rng(0)

    def predict(self, *, observation: VLAObservation) -> VLAActionChunk:
        t = np.arange(self.horizon)[:, None]
        base = np.sin(0.2 * t + np.array([0.0, 1.0, 2.0])[None, :])
        offset = self._rng.normal(0.0, self.mode_strength, size=3)
        chunk = (base + offset[None, :]).astype(np.float32)
        return VLAActionChunk(actions=[VLAAction(data=row, spec=_SPEC) for row in chunk])


# ---- variable-delay emission ------------------------------------------------------------


def test_variable_emission_matches_constant_when_delays_equal() -> None:
    chunks = [np.random.default_rng(i).normal(size=(12, 3)).astype(np.float32) for i in range(5)]
    delays = [3] * 5
    var, late = simulate_emission_variable(
        chunks, delays, horizon=12, execute_horizon=6, freeze=False
    )
    assert late == 0
    assert var.emitted.shape[0] == 6 * 5


def test_variable_emission_flags_late_cycles() -> None:
    chunks = [np.zeros((10, 3), dtype=np.float32) for _ in range(3)]
    # delay 8 + execute 4 > horizon 10 -> clamped, counted late (cycles 1 and 2)
    _, late = simulate_emission_variable(
        chunks, [0, 8, 8], horizon=10, execute_horizon=4, freeze=False
    )
    assert late == 2


def test_variable_emission_rejects_mismatched_delays() -> None:
    chunks = [np.zeros((8, 3), dtype=np.float32) for _ in range(2)]
    with pytest.raises(ValueError, match="one entry per chunk"):
        simulate_emission_variable(chunks, [1], horizon=8, execute_horizon=4, freeze=False)


# ---- recorder ---------------------------------------------------------------------------


def test_record_rtc_trace_with_injected_clock_is_deterministic() -> None:
    adapter = _FakeChunkAdapter(horizon=16)
    obs = [VLAObservation(instruction="pick") for _ in range(6)]
    # injected clock: each predict takes 0.12 s (start, stop alternate)
    ticks = iter([v for i in range(6) for v in (i * 1.0, i * 1.0 + 0.12)])
    trace = record_rtc_trace(
        adapter, obs, control_hz=30.0, execute_horizon=8, clock=lambda: next(ticks)
    )
    assert trace.horizon == 16
    assert len(trace.events) == 6
    assert all(abs(e.inference_latency_ms - 120.0) < 1e-6 for e in trace.events)
    # 0.12 s * 30 Hz = 3.6 -> rounds to 4 ticks
    assert trace_delays_ticks(trace)[1] == 4


def test_record_rtc_trace_rejects_non_chunk_adapter() -> None:
    class _SingleAction:
        name = "single"

        def predict(self, *, observation: VLAObservation) -> VLAAction:
            return VLAAction(data=np.zeros(3, dtype=np.float32), spec=_SPEC)

    with pytest.raises(TypeError, match="VLAActionChunk"):
        record_rtc_trace(_SingleAction(), [VLAObservation(instruction="x")])


def test_build_trace_rejects_ragged_chunks() -> None:
    with pytest.raises(ValueError, match="expected"):
        build_trace_from_chunks(
            [([[0.0, 0.0]], 10.0), ([[0.0, 0.0], [1.0, 1.0]], 10.0)],
            model="m",
            control_hz=30.0,
            execute_horizon=1,
            source="t",
        )


# ---- trace round-trip + comparison ------------------------------------------------------


def test_trace_to_from_dict_round_trips() -> None:
    adapter = _FakeChunkAdapter(horizon=12)
    obs = [VLAObservation(instruction="pick") for _ in range(4)]
    ticks = iter([v for i in range(4) for v in (i * 1.0, i * 1.0 + 0.05)])
    trace = record_rtc_trace(adapter, obs, execute_horizon=6, clock=lambda: next(ticks))
    payload = trace.to_dict()
    assert payload["schema_version"] == RTC_TRACE_SCHEMA_VERSION
    restored = RTCTrace.from_dict(payload)
    assert restored.horizon == trace.horizon
    assert restored.events[2].chunk == trace.events[2].chunk


def test_compare_trace_strategies_reduces_boundary_jump() -> None:
    adapter = _FakeChunkAdapter(horizon=16, mode_strength=0.8)
    obs = [VLAObservation(instruction="pick") for _ in range(10)]
    ticks = iter([v for i in range(10) for v in (i * 1.0, i * 1.0 + 0.13)])
    trace = record_rtc_trace(
        adapter, obs, control_hz=30.0, execute_horizon=8, clock=lambda: next(ticks)
    )
    report = compare_trace_strategies(trace)
    payload = report.to_dict()
    assert payload["schema_version"] == RTC_SIM_SCHEMA_VERSION
    assert report.rtc.mean_boundary_jump < report.naive.mean_boundary_jump
    assert "real per-cycle delays" in payload["source"]
