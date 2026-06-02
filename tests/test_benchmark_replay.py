from __future__ import annotations

from pathlib import Path

from vla_zoo.benchmark.replay import (
    REPLAY_SOURCE,
    ROSBAG_REPLAY_NOTE,
    frames_to_records,
    load_action_log,
    replay_action_rate_hz,
)
from vla_zoo.benchmark.results import RESULT_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_LOG = REPO_ROOT / "docs" / "assets" / "sample_ros2_remote_dummy" / "vla_actions.jsonl"


def test_rosbag_replay_note_is_honest_about_scope() -> None:
    assert "rosbag2" in ROSBAG_REPLAY_NOTE
    assert "not yet implemented" in ROSBAG_REPLAY_NOTE
    assert "No task-success claim" in ROSBAG_REPLAY_NOTE


def test_load_action_log_parses_sample_frames() -> None:
    frames = load_action_log(SAMPLE_LOG)

    assert frames, "sample action log should contain frames"
    first = frames[0]
    assert first.model_name == "dummy"
    assert first.action_space == "eef_delta"
    assert len(first.action) == 7
    assert first.stamp_sec > 0
    assert first.latency_ms is not None


def test_replay_action_rate_is_positive_for_sample() -> None:
    frames = load_action_log(SAMPLE_LOG)
    rate = replay_action_rate_hz(frames)

    assert rate is not None
    assert rate > 0


def test_frames_to_records_make_no_success_claim() -> None:
    frames = load_action_log(SAMPLE_LOG)
    records = frames_to_records(frames)

    assert len(records) == len(frames)
    assert all(r.success is None for r in records)
    assert all(r.source == REPLAY_SOURCE for r in records)
    assert all(r.schema_version == RESULT_SCHEMA_VERSION for r in records)


def test_replay_sample_summary_artifact_exists() -> None:
    assert (REPO_ROOT / "docs" / "assets" / "sample_benchmark" / "ros2_replay_summary.md").is_file()
    assert (REPO_ROOT / "docs" / "benchmark_results.md").is_file()
