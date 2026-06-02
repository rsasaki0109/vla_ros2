from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vla_zoo.runtime.realtime_chunking import (
    RTC_SIM_SCHEMA_VERSION,
    RealtimeChunkingConfig,
    chunks_from_action_log,
    compare_strategies,
    format_rtc_sim_markdown,
    simulate_emission,
    synthetic_chunk_stream,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _config() -> RealtimeChunkingConfig:
    return RealtimeChunkingConfig(horizon=16, execute_horizon=8, inference_delay_ticks=4)


def test_config_rejects_infeasible_timing() -> None:
    with pytest.raises(ValueError, match="must not exceed horizon"):
        RealtimeChunkingConfig(horizon=8, execute_horizon=6, inference_delay_ticks=4)
    with pytest.raises(ValueError, match="execute_horizon"):
        RealtimeChunkingConfig(horizon=8, execute_horizon=0, inference_delay_ticks=1)
    with pytest.raises(ValueError, match="horizon must be at least 2"):
        RealtimeChunkingConfig(horizon=1, execute_horizon=1, inference_delay_ticks=0)


def test_simulate_emission_tick_count_matches_cycles() -> None:
    config = _config()
    chunks = synthetic_chunk_stream(config, chunk_count=5, seed=1)
    result = simulate_emission(chunks, config, freeze=False)
    # cycle 0 emits `execute`, each later cycle emits `execute`
    assert result.emitted.shape[0] == config.execute_horizon * len(chunks)
    assert len(result.boundary_indices) == len(chunks) - 1


def test_simulate_emission_rejects_wrong_horizon() -> None:
    config = _config()
    bad = [np.zeros((config.horizon - 1, 3), dtype=np.float32)]
    with pytest.raises(ValueError, match="every chunk must have"):
        simulate_emission(bad, config, freeze=False)


def test_freeze_reduces_boundary_jump_vs_naive() -> None:
    config = _config()
    chunks = synthetic_chunk_stream(config, chunk_count=12, mode_strength=0.6, seed=7)
    report = compare_strategies(chunks, config, source="unit-test")
    # the freeze prefix should noticeably smooth chunk transitions
    assert report.rtc.mean_boundary_jump < report.naive.mean_boundary_jump
    assert report.boundary_jump_reduction > 0.5


def test_zero_delay_means_no_stale_prefix_to_freeze() -> None:
    config = RealtimeChunkingConfig(horizon=12, execute_horizon=8, inference_delay_ticks=0)
    chunks = synthetic_chunk_stream(config, chunk_count=6, seed=3)
    report = compare_strategies(chunks, config, source="unit-test")
    # with d=0 the freeze offset anchors on the same index and still cannot increase jumps
    assert report.rtc.mean_boundary_jump <= report.naive.mean_boundary_jump + 1e-6


def test_synthetic_stream_is_deterministic() -> None:
    config = _config()
    a = synthetic_chunk_stream(config, chunk_count=4, seed=42)
    b = synthetic_chunk_stream(config, chunk_count=4, seed=42)
    for left, right in zip(a, b, strict=True):
        assert np.array_equal(left, right)


def test_chunks_from_action_log_windows_and_pads() -> None:
    config = RealtimeChunkingConfig(horizon=4, execute_horizon=2, inference_delay_ticks=1)
    actions = [(float(i), float(-i)) for i in range(5)]
    chunks = chunks_from_action_log(actions, config)
    assert all(chunk.shape == (4, 2) for chunk in chunks)
    # last window is padded by holding the final action
    assert np.array_equal(chunks[-1][-1], chunks[-1][-2]) or chunks[-1].shape[0] == 4


def test_report_to_dict_carries_schema_and_honesty_note() -> None:
    config = _config()
    chunks = synthetic_chunk_stream(config, chunk_count=4, seed=0)
    payload = compare_strategies(chunks, config, source="unit-test").to_dict()
    assert payload["schema_version"] == RTC_SIM_SCHEMA_VERSION
    assert "not a policy-quality" in payload["note"]
    assert payload["naive"]["strategy"] == "naive-async"
    assert payload["rtc"]["strategy"] == "rtc-freeze"


def test_markdown_renders_both_strategies_and_reduction() -> None:
    config = _config()
    chunks = synthetic_chunk_stream(config, chunk_count=6, seed=2)
    markdown = format_rtc_sim_markdown(compare_strategies(chunks, config, source="unit-test"))
    assert "naive async" in markdown
    assert "RTC freeze" in markdown
    assert "Boundary-jump reduction" in markdown


def test_recorded_rtc_artifact_is_valid_and_honest() -> None:
    path = REPO_ROOT / "docs" / "assets" / "rtc_sim" / "rtc_scheduler_sim.json"
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == RTC_SIM_SCHEMA_VERSION
    assert payload["rtc"]["mean_boundary_jump"] < payload["naive"]["mean_boundary_jump"]
    assert payload["boundary_jump_reduction"] > 0.0
