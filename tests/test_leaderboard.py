from __future__ import annotations

from pathlib import Path

import pytest

from vla_zoo.benchmark.results import BenchmarkSummary
from vla_zoo.compare.leaderboard import (
    LEADERBOARD_SCHEMA_VERSION,
    RUNTIME_PROFILES,
    RuntimeProfile,
    build_leaderboard,
    format_leaderboard_html,
    format_leaderboard_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _summary(model: str, *, p50: float | None) -> BenchmarkSummary:
    return BenchmarkSummary(
        model=model,
        source="pybullet-action-probe",
        sample_count=21,
        success_count=0,
        success_rate=None,
        latency_ms_p50=p50,
        latency_ms_p95=None if p50 is None else p50 * 1.5,
        latency_ms_mean=p50,
        action_rate_hz=1.0,
        exception_count=0,
    )


def test_leaderboard_ranks_measured_and_appends_blocked() -> None:
    board = build_leaderboard(
        [_summary("openvla", p50=2000.0), _summary("smolvla", p50=382.0)],
        metric="latency_ms_p50",
    )
    by_model = {e.model: e for e in board.entries}
    # measured adapters are ranked by latency
    assert by_model["smolvla"].rank == 1
    assert by_model["openvla"].rank == 2
    # blocked adapters with no measured run are appended unranked, not dropped
    assert by_model["pi0"].rank is None
    assert by_model["pi0"].status == "blocked"
    assert by_model["pi0"].latency_ms_p50 is None  # no fabricated latency
    assert by_model["pi0"].memory_gb == RUNTIME_PROFILES["pi0"].memory_gb
    assert by_model["groot"].rank is None


def test_leaderboard_joins_recorded_profile() -> None:
    board = build_leaderboard([_summary("smolvla", p50=382.0)], metric="latency_ms_p50")
    smolvla = next(e for e in board.entries if e.model == "smolvla")
    assert smolvla.status == "verified"
    assert smolvla.memory_gb == 0.97
    assert smolvla.evidence_link is not None


def test_leaderboard_no_blocked_omits_unmeasured() -> None:
    board = build_leaderboard(
        [_summary("smolvla", p50=382.0)],
        metric="latency_ms_p50",
        include_blocked=False,
    )
    assert {e.model for e in board.entries} == {"smolvla"}


def test_leaderboard_unknown_model_is_measured_status() -> None:
    board = build_leaderboard(
        [_summary("mystery", p50=100.0)],
        metric="latency_ms_p50",
        profiles={},
    )
    entry = next(e for e in board.entries if e.model == "mystery")
    assert entry.status == "measured"
    assert entry.memory_gb is None


def test_leaderboard_unsupported_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        build_leaderboard([_summary("a", p50=1.0)], metric="bogus")


def test_leaderboard_to_dict_is_schema_versioned() -> None:
    board = build_leaderboard([_summary("smolvla", p50=382.0)], metric="latency_ms_p50")
    payload = board.to_dict()
    assert payload["schema_version"] == LEADERBOARD_SCHEMA_VERSION
    assert payload["metric"] == "latency_ms_p50"
    assert payload["entries"][0]["model"] == "smolvla"


def test_leaderboard_markdown_is_runtime_centric() -> None:
    board = build_leaderboard([_summary("smolvla", p50=382.0)], metric="latency_ms_p50")
    md = format_leaderboard_markdown(board)
    assert "VLA Runtime Leaderboard" in md
    assert "NOT by robot task-success" in md
    assert "🥇" in md  # medal for rank 1


def test_leaderboard_html_renders_badges_and_disclaimer() -> None:
    board = build_leaderboard(
        [_summary("openvla", p50=2000.0), _summary("smolvla", p50=382.0)],
        metric="latency_ms_p50",
    )
    html = format_leaderboard_html(board)
    assert "<!doctype html>" in html
    assert "badge ok" in html  # verified -> ok badge
    assert "badge bad" in html  # blocked -> bad badge
    assert "Not</strong> a robot task-success" in html


def test_recorded_leaderboard_artifacts_exist() -> None:
    base = REPO_ROOT / "docs" / "assets" / "leaderboard"
    assert (base / "vla_runtime_leaderboard.json").is_file()
    assert (base / "vla_runtime_leaderboard.md").is_file()
    assert (base / "vla_runtime_leaderboard.html").is_file()


def test_runtime_profiles_are_well_formed() -> None:
    for model, profile in RUNTIME_PROFILES.items():
        assert isinstance(profile, RuntimeProfile)
        assert profile.status in {"verified", "partial", "blocked", "planned"}
        # blocked-without-memory is allowed, but a stated memory must be positive
        if profile.memory_gb is not None:
            assert profile.memory_gb > 0
        assert profile.note, f"{model} profile needs a note"
