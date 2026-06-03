from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vla_zoo.benchmark.replay import frames_to_records, load_action_log
from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk
from vla_zoo.demo.action_probe import (
    PROBE_IMAGE_SOURCE,
    build_probe_record,
    format_action_probe_summary_markdown,
    summarize_probe_records,
    write_action_probe_log,
)
from vla_zoo.demo.pybullet import AdapterPredictionEvent


def _spec() -> ActionSpec:
    return ActionSpec(
        action_space="custom",
        shape=(6,),
        names=("a", "b", "c", "d", "e", "f"),
    )


def _action(values: list[float]) -> VLAAction:
    return VLAAction(data=np.asarray(values, dtype=np.float32), spec=_spec())


def _event(
    *,
    frame_index: int,
    query_index: int,
    sim_time: float,
    latency_ms: float | None,
    prediction: VLAAction | VLAActionChunk,
) -> AdapterPredictionEvent:
    return AdapterPredictionEvent(
        frame_index=frame_index,
        query_index=query_index,
        sim_time=sim_time,
        phase="approach",
        model_name="smolvla",
        runtime="local",
        instruction="pick up the red block",
        task_id="pick_red_block",
        prediction=prediction,
        latency_ms=latency_ms,
        cube_position=(0.58, -0.16, 0.035),
        cube_goal_position=(0.58, 0.22, 0.035),
        camera_keys=("primary", "observation.images.camera1"),
    )


def test_build_probe_record_captures_full_action_and_marks_real_scene() -> None:
    event = _event(
        frame_index=4,
        query_index=2,
        sim_time=1.5,
        latency_ms=95.0,
        prediction=_action([0.2, -0.4, 0.0, 0.1, 0.0, -0.6]),
    )

    record = build_probe_record(event)

    # full 6D action is recorded, not the 4D demo overlay
    assert record["data"] == pytest.approx([0.2, -0.4, 0.0, 0.1, 0.0, -0.6])
    assert record["names"] == ["a", "b", "c", "d", "e", "f"]
    assert record["model_name"] == "smolvla"
    assert record["header"]["stamp"] == {"sec": 1, "nanosec": 500_000_000}
    meta = record["metadata"]
    assert meta["latency_ms"] == 95.0
    assert meta["image_source"] == PROBE_IMAGE_SOURCE
    assert meta["real_scene"] is True
    assert meta["policy_quality"] == "not_verified"
    assert meta["abs_action_max"] == pytest.approx(0.6)
    assert meta["abs_action_mean"] == pytest.approx((0.2 + 0.4 + 0.0 + 0.1 + 0.0 + 0.6) / 6)


def test_build_probe_record_uses_first_action_of_chunk() -> None:
    chunk = VLAActionChunk(actions=[_action([1.0, 0, 0, 0, 0, 0]), _action([2.0, 0, 0, 0, 0, 0])])
    event = _event(
        frame_index=0, query_index=1, sim_time=0.0, latency_ms=10.0, prediction=chunk
    )

    record = build_probe_record(event)

    assert record["data"][0] == pytest.approx(1.0)
    assert record["metadata"]["chunk_size"] == 2


def test_summarize_probe_records_reduces_latency_and_magnitude() -> None:
    records = [
        build_probe_record(
            _event(
                frame_index=i,
                query_index=i + 1,
                sim_time=float(i),
                latency_ms=latency,
                prediction=_action([mag, 0, 0, 0, 0, 0]),
            )
        )
        for i, (latency, mag) in enumerate([(100.0, 0.2), (300.0, 0.6), (200.0, 0.4)])
    ]

    summary = summarize_probe_records(
        records,
        model="smolvla",
        runtime="local",
        instruction="pick up the red block",
        task_id="pick_red_block",
    )

    assert summary.record_count == 3
    assert summary.action_dim == 6
    assert summary.latency_ms_min == 100.0
    assert summary.latency_ms_max == 300.0
    assert summary.latency_ms_p50 == 200.0
    assert summary.abs_action_max == pytest.approx(0.6)
    assert summary.policy_quality == "not_verified"
    assert summary.image_source == PROBE_IMAGE_SOURCE


def test_summarize_probe_records_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarize_probe_records(
            [], model="smolvla", runtime="local", instruction="x", task_id="t"
        )


def test_summary_markdown_is_runtime_centric() -> None:
    summary = summarize_probe_records(
        [
            build_probe_record(
                _event(
                    frame_index=0,
                    query_index=1,
                    sim_time=0.0,
                    latency_ms=120.0,
                    prediction=_action([0.1, 0, 0, 0, 0, 0]),
                )
            )
        ],
        model="smolvla",
        runtime="local",
        instruction="pick up the red block",
        task_id="pick_red_block",
    )

    text = format_action_probe_summary_markdown(summary)

    assert "Real-Scene Action Probe" in text
    assert "not_verified" in text
    assert "NOT a task-success or policy-quality claim" in text
    assert PROBE_IMAGE_SOURCE in text


def test_written_log_round_trips_through_action_replay(tmp_path: Path) -> None:
    records = [
        build_probe_record(
            _event(
                frame_index=i,
                query_index=i + 1,
                sim_time=float(i) + 0.25,
                latency_ms=100.0 + i,
                prediction=_action([0.1 * i, 0, 0, 0, 0, 0]),
            )
        )
        for i in range(3)
    ]

    path = write_action_probe_log(tmp_path / "probe.jsonl", records)
    frames = load_action_log(path)

    assert len(frames) == 3
    assert frames[0].model_name == "smolvla"
    assert frames[1].latency_ms == 101.0
    assert frames[2].stamp_sec == pytest.approx(2.25)
    # the replay path always records success=None -- no task-success claim
    episodes = frames_to_records(frames)
    assert all(episode.success is None for episode in episodes)
