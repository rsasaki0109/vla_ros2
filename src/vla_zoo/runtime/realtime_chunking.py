"""Latency-aware action-chunk scheduling simulation (Real-Time Chunking style).

Action-chunking policies predict a horizon of ``H`` future actions per inference. A
controller commits to executing some of them while the next chunk is computed in the
background. Because that inference takes time, by the moment the new chunk arrives its
first few actions describe timesteps that have *already* been executed -- and since the
new chunk is an independent sample, naively switching to it produces a discontinuity at
the chunk boundary (the policy "jumps" between incompatible plans).

Real-Time Chunking (RTC, Black et al., arXiv:2506.07339) addresses this purely at
inference time, with no retraining: the actions guaranteed to execute before the new
chunk arrives are "frozen" to the values the previous chunk already committed, and the
remainder is "inpainted" to stay consistent with that frozen prefix.

This module is a **pure, deterministic simulation** of the *scheduling* layer of that
idea. It does not run a diffusion/flow policy and does not implement the gradient-guided
sampler (which needs the model's velocity field and backprop). Instead it models the
freeze + soft-mask blend over a stream of pre-computed chunks so the boundary-continuity
behaviour can be measured and compared on CPU without any model. It makes **no
policy-quality or task-success claim**: it characterises a runtime scheduling mechanism,
not how good any policy's actions are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Schema identifier for the JSON artifact this module emits.
RTC_SIM_SCHEMA_VERSION = "vla-zoo-rtc-sim/v1"

#: One reusable disclaimer kept verbatim across renderers.
RTC_SIM_NOTE = (
    "Deterministic simulation of latency-aware action-chunk scheduling (Real-Time "
    "Chunking style). Models the freeze-prefix + soft-mask blend over pre-computed "
    "chunks; it does not run a diffusion/flow policy or the gradient-guided sampler. "
    "Continuity is a runtime scheduling property, not a policy-quality or task-success "
    "claim."
)


@dataclass(frozen=True)
class RealtimeChunkingConfig:
    """Timing parameters for the chunk scheduler.

    ``horizon`` is the number of actions a single inference predicts (``H``).
    ``execute_horizon`` is how many actions the controller commits to executing per
    cycle before swapping to the next chunk (``s``). ``inference_delay_ticks`` is how
    many control ticks one inference takes to complete (``d``); the new chunk's first
    ``d`` actions therefore land on already-executed timesteps.
    """

    horizon: int
    execute_horizon: int
    inference_delay_ticks: int
    control_hz: float = 50.0

    def __post_init__(self) -> None:
        if self.horizon < 2:
            msg = "horizon must be at least 2"
            raise ValueError(msg)
        if not 1 <= self.execute_horizon <= self.horizon:
            msg = "execute_horizon must be in [1, horizon]"
            raise ValueError(msg)
        if self.inference_delay_ticks < 0:
            msg = "inference_delay_ticks must be non-negative"
            raise ValueError(msg)
        if self.inference_delay_ticks + self.execute_horizon > self.horizon:
            msg = "inference_delay_ticks + execute_horizon must not exceed horizon"
            raise ValueError(msg)
        if self.control_hz <= 0:
            msg = "control_hz must be positive"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "execute_horizon": self.execute_horizon,
            "inference_delay_ticks": self.inference_delay_ticks,
            "control_hz": self.control_hz,
        }


@dataclass(frozen=True)
class EmissionResult:
    """The control stream produced by one scheduling strategy plus its continuity stats."""

    strategy: str
    emitted: NDArray[np.float32]
    boundary_indices: tuple[int, ...]
    mean_boundary_jump: float
    max_boundary_jump: float
    mean_step_jump: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "tick_count": int(self.emitted.shape[0]),
            "boundary_count": len(self.boundary_indices),
            "mean_boundary_jump": self.mean_boundary_jump,
            "max_boundary_jump": self.max_boundary_jump,
            "mean_step_jump": self.mean_step_jump,
        }


@dataclass(frozen=True)
class RTCSimReport:
    """Side-by-side comparison of naive-async vs RTC freeze scheduling."""

    config: RealtimeChunkingConfig
    chunk_count: int
    action_dim: int
    naive: EmissionResult
    rtc: EmissionResult
    boundary_jump_reduction: float
    source: str
    note: str = RTC_SIM_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RTC_SIM_SCHEMA_VERSION,
            "source": self.source,
            "config": self.config.to_dict(),
            "chunk_count": self.chunk_count,
            "action_dim": self.action_dim,
            "naive": self.naive.to_dict(),
            "rtc": self.rtc.to_dict(),
            "boundary_jump_reduction": self.boundary_jump_reduction,
            "note": self.note,
        }


def _soft_weight(chunk_index: int, delay: int, horizon: int) -> float:
    """Soft-mask weight for the offset correction at a given chunk index.

    Weight is 1.0 at the first executable index (``delay``) so the new chunk connects
    smoothly to the frozen prefix, then decays exponentially toward 0 so far-future
    actions return to the raw predicted chunk. Mirrors the soft mask in RTC's Eq. 5
    with a deterministic offset blend in place of the gradient-guided sampler.
    """

    distance = chunk_index - delay
    if distance < 0:
        return 1.0
    tau = max(1.0, (horizon - delay) / 3.0)
    return float(np.exp(-distance / tau))


def _continuity_stats(
    emitted: list[NDArray[np.float32]], boundaries: list[int]
) -> tuple[float, float, float]:
    if len(emitted) < 2:
        return 0.0, 0.0, 0.0
    def jump(left: int, right: int) -> float:
        return float(np.linalg.norm(emitted[right] - emitted[left]))

    step_jumps = [jump(t - 1, t) for t in range(1, len(emitted))]
    boundary_jumps = [jump(b - 1, b) for b in boundaries if b >= 1]
    mean_boundary = float(np.mean(boundary_jumps)) if boundary_jumps else 0.0
    max_boundary = float(np.max(boundary_jumps)) if boundary_jumps else 0.0
    mean_step = float(np.mean(step_jumps))
    return mean_boundary, max_boundary, mean_step


def _emit_with_delays(
    arrays: list[NDArray[np.float32]],
    delays: list[int],
    *,
    horizon: int,
    execute: int,
    freeze: bool,
) -> tuple[EmissionResult, int]:
    """Core emission shared by the constant- and variable-delay paths.

    ``delays[k]`` is the inference delay (in control ticks) for the chunk that takes over
    at cycle ``k`` (``delays[0]`` is unused). A delay too large to leave ``execute``
    actions in the horizon means the chunk arrived late; it is clamped and counted.
    """

    emitted: list[NDArray[np.float32]] = []
    boundaries: list[int] = []
    prev_last: NDArray[np.float32] | None = None
    late = 0
    for cycle, chunk in enumerate(arrays):
        if cycle == 0:
            segment = chunk[0:execute].copy()
        else:
            delay = delays[cycle]
            if delay + execute > horizon:  # chunk arrived too late to cover the cycle
                delay = max(0, horizon - execute)
                late += 1
            segment = chunk[delay : delay + execute].copy()
            # With delay == 0 there is no stale prefix to freeze (inference is
            # instantaneous), so the freeze strategy degenerates to naive async.
            if freeze and prev_last is not None and delay >= 1:
                offset = prev_last - chunk[delay - 1]
                for local, chunk_index in enumerate(range(delay, delay + execute)):
                    weight = _soft_weight(chunk_index, delay, horizon)
                    segment[local] = chunk[chunk_index] + weight * offset
            boundaries.append(len(emitted))
        emitted.extend(segment)
        prev_last = emitted[-1]

    mean_boundary, max_boundary, mean_step = _continuity_stats(emitted, boundaries)
    result = EmissionResult(
        strategy="rtc-freeze" if freeze else "naive-async",
        emitted=np.stack(emitted).astype(np.float32),
        boundary_indices=tuple(boundaries),
        mean_boundary_jump=mean_boundary,
        max_boundary_jump=max_boundary,
        mean_step_jump=mean_step,
    )
    return result, late


def _validated_arrays(
    chunks: list[NDArray[np.float32]], horizon: int
) -> list[NDArray[np.float32]]:
    if not chunks:
        msg = "at least one chunk is required"
        raise ValueError(msg)
    arrays = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
    for chunk in arrays:
        if chunk.shape[0] != horizon:
            msg = f"every chunk must have {horizon} actions, got {chunk.shape[0]}"
            raise ValueError(msg)
    return arrays


def simulate_emission(
    chunks: list[NDArray[np.float32]], config: RealtimeChunkingConfig, *, freeze: bool
) -> EmissionResult:
    """Roll the pre-computed chunks out into a control stream under one strategy.

    With ``freeze=False`` (naive async) each new chunk is executed raw from its
    current-time index ``d``, so the boundary jump exposes the inter-chunk mode change.
    With ``freeze=True`` (RTC) the new chunk's executable segment is offset-corrected so
    it connects to the previous chunk's last committed action, with an exponentially
    decaying soft mask returning later actions to the raw prediction.
    """

    arrays = _validated_arrays(chunks, config.horizon)
    delays = [config.inference_delay_ticks] * len(arrays)
    result, _ = _emit_with_delays(
        arrays, delays, horizon=config.horizon, execute=config.execute_horizon, freeze=freeze
    )
    return result


def simulate_emission_variable(
    chunks: list[NDArray[np.float32]],
    delays_ticks: list[int],
    *,
    horizon: int,
    execute_horizon: int,
    freeze: bool,
) -> tuple[EmissionResult, int]:
    """Emit under per-cycle inference delays (e.g. recorded from a real model run).

    Returns the emission and the number of late cycles (a chunk whose measured delay left
    fewer than ``execute_horizon`` fresh actions in the horizon). Unlike the constant-delay
    path this models the variable inference latency a real adapter exhibits.
    """

    if len(delays_ticks) != len(chunks):
        msg = "delays_ticks must have one entry per chunk"
        raise ValueError(msg)
    if not 1 <= execute_horizon <= horizon:
        msg = "execute_horizon must be in [1, horizon]"
        raise ValueError(msg)
    arrays = _validated_arrays(chunks, horizon)
    clamped = [max(0, int(d)) for d in delays_ticks]
    return _emit_with_delays(
        arrays, clamped, horizon=horizon, execute=execute_horizon, freeze=freeze
    )


def compare_strategies(
    chunks: list[NDArray[np.float32]],
    config: RealtimeChunkingConfig,
    *,
    source: str,
) -> RTCSimReport:
    """Simulate both strategies over the same chunk stream and report the reduction."""

    naive = simulate_emission(chunks, config, freeze=False)
    rtc = simulate_emission(chunks, config, freeze=True)
    if naive.mean_boundary_jump > 0:
        reduction = (naive.mean_boundary_jump - rtc.mean_boundary_jump) / naive.mean_boundary_jump
    else:
        reduction = 0.0
    return RTCSimReport(
        config=config,
        chunk_count=len(chunks),
        action_dim=int(np.asarray(chunks[0]).shape[1]),
        naive=naive,
        rtc=rtc,
        boundary_jump_reduction=float(reduction),
        source=source,
    )


def synthetic_chunk_stream(
    config: RealtimeChunkingConfig,
    *,
    chunk_count: int,
    action_dim: int = 3,
    mode_strength: float = 0.6,
    jitter: float = 0.02,
    seed: int = 0,
) -> list[NDArray[np.float32]]:
    """Build a deterministic synthetic chunk stream that exhibits inter-chunk mode jumps.

    A smooth shared reference trajectory is sampled at each re-plan; every chunk also
    carries a per-chunk constant offset (the "mode" the policy committed to that cycle)
    plus light per-index jitter. The per-chunk offset is what a naive async swap surfaces
    as a boundary discontinuity -- it is a synthetic stress input for the scheduler, not
    recorded model output.
    """

    if chunk_count < 1:
        msg = "chunk_count must be positive"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    execute = config.execute_horizon
    phase = rng.uniform(0.0, np.pi, size=action_dim)
    freq = np.linspace(0.18, 0.32, action_dim)

    def reference(tick: int) -> NDArray[np.float32]:
        t = np.arange(tick, tick + config.horizon)[:, None]
        wave = np.sin(freq[None, :] * t + phase[None, :])
        return wave.astype(np.float32)

    chunks: list[NDArray[np.float32]] = []
    for cycle in range(chunk_count):
        base = reference(cycle * execute)
        mode_offset = rng.normal(0.0, mode_strength, size=action_dim).astype(np.float32)
        noise = rng.normal(0.0, jitter, size=base.shape).astype(np.float32)
        chunks.append(base + mode_offset[None, :] + noise)
    return chunks


def chunks_from_action_log(
    actions: list[tuple[float, ...]], config: RealtimeChunkingConfig
) -> list[NDArray[np.float32]]:
    """Window a recorded action stream into overlapping horizon-length chunks.

    Re-plans every ``execute_horizon`` steps with a ``horizon``-length lookahead, padding
    the final window by holding the last action. Useful for driving the scheduler from a
    real ``vla_actions.jsonl`` stream; note a consistent recorded stream shows small
    boundary jumps because adjacent windows agree.
    """

    if not actions:
        msg = "at least one action is required"
        raise ValueError(msg)
    matrix = np.asarray(actions, dtype=np.float32)
    if matrix.ndim != 2:
        msg = "actions must be a list of equal-length vectors"
        raise ValueError(msg)
    horizon, execute = config.horizon, config.execute_horizon
    chunks: list[NDArray[np.float32]] = []
    start = 0
    while start < matrix.shape[0]:
        window = matrix[start : start + horizon]
        if window.shape[0] < horizon:
            pad = np.repeat(window[-1:], horizon - window.shape[0], axis=0)
            window = np.concatenate([window, pad], axis=0)
        chunks.append(window)
        start += execute
    return chunks


def format_rtc_sim_markdown(report: RTCSimReport) -> str:
    """Render the comparison as a runtime-centric Markdown report."""

    config = report.config
    lines = [
        "# Real-Time Chunking scheduler simulation",
        "",
        f"- Source: `{report.source}`",
        f"- Chunks: {report.chunk_count} (horizon {config.horizon}, "
        f"execute {config.execute_horizon}, inference delay {config.inference_delay_ticks} ticks)",
        f"- Control rate: {config.control_hz:g} Hz, action dim {report.action_dim}",
        "",
        "| Strategy | Mean boundary jump | Max boundary jump | Mean step jump |",
        "|---|---:|---:|---:|",
        f"| naive async | {report.naive.mean_boundary_jump:.4f} "
        f"| {report.naive.max_boundary_jump:.4f} | {report.naive.mean_step_jump:.4f} |",
        f"| RTC freeze | {report.rtc.mean_boundary_jump:.4f} "
        f"| {report.rtc.max_boundary_jump:.4f} | {report.rtc.mean_step_jump:.4f} |",
        "",
        f"**Boundary-jump reduction: {report.boundary_jump_reduction * 100:.1f}%** "
        "(lower boundary jump means smoother chunk transitions).",
        "",
        f"> {report.note}",
        "",
    ]
    return "\n".join(lines)
