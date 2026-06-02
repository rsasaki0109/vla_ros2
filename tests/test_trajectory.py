from __future__ import annotations

from pathlib import Path

import pytest

from vla_zoo.benchmark.replay import ReplayFrame
from vla_zoo.demo.trajectory import (
    Trajectory,
    build_trajectory,
    render_trajectory_gif,
    render_trajectory_race_gif,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _frame(index: int, action: tuple[float, ...]) -> ReplayFrame:
    return ReplayFrame(
        index=index,
        stamp_sec=float(index),
        model_name="demo",
        action_space="eef_delta",
        action=action,
        names=(),
        latency_ms=10.0,
    )


def test_build_trajectory_integrates_position_deltas() -> None:
    frames = [
        _frame(0, (1.0, 0.0, 0.0, 0, 0, 0, 1.0)),
        _frame(1, (0.0, 2.0, 0.0, 0, 0, 0, 0.0)),
        _frame(2, (0.0, 0.0, 3.0, 0, 0, 0, 1.0)),
    ]
    traj = build_trajectory(frames)

    assert traj.points[0] == (0.0, 0.0, 0.0)  # starts at origin
    assert traj.points[1] == (1.0, 0.0, 0.0)
    assert traj.points[2] == (1.0, 2.0, 0.0)  # cumulative
    assert traj.points[3] == (1.0, 2.0, 3.0)
    assert traj.step_count == 3
    # the 7th dim is captured as the gripper signal
    assert traj.gripper[1] == 1.0
    assert traj.gripper[2] == 0.0


def test_build_trajectory_scale_multiplies_deltas() -> None:
    traj = build_trajectory([_frame(0, (2.0, 0.0, 0.0))], scale=0.5)
    assert traj.points[1][0] == 1.0


def test_build_trajectory_without_gripper_dim_is_none() -> None:
    traj = build_trajectory([_frame(0, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))])  # 6-DoF
    assert traj.gripper[1] is None


def test_build_trajectory_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty action log"):
        build_trajectory([])


def test_trajectory_to_dict_round_trips_shape() -> None:
    traj = build_trajectory([_frame(0, (1.0, 1.0, 1.0, 0, 0, 0, 1.0))])
    payload = traj.to_dict()
    assert payload["model"] == "demo"
    assert payload["step_count"] == 1
    assert payload["points"][0] == [0.0, 0.0, 0.0]


def test_render_trajectory_gif_writes_animated_gif(tmp_path: Path) -> None:
    from PIL import Image

    frames = [
        _frame(0, (0.5, 0.2, -0.1, 0, 0, 0, 1.0)),
        _frame(1, (0.3, -0.4, 0.2, 0, 0, 0, 0.0)),
        _frame(2, (-0.2, 0.1, 0.3, 0, 0, 0, 1.0)),
    ]
    traj = build_trajectory(frames)
    out = tmp_path / "traj.gif"
    render_trajectory_gif(traj, out, width=520)

    assert out.is_file()
    with Image.open(out) as img:
        assert img.format == "GIF"
        assert img.is_animated
        assert img.n_frames == traj.step_count
        assert img.size[0] == 520


def test_render_trajectory_gif_needs_two_points(tmp_path: Path) -> None:
    traj = Trajectory(
        model="x", action_space="eef_delta", points=((0.0, 0.0, 0.0),), gripper=(None,)
    )
    with pytest.raises(ValueError, match="at least two points"):
        render_trajectory_gif(traj, tmp_path / "x.gif")


def test_render_trajectory_race_overlays_and_runs_to_longest(tmp_path: Path) -> None:
    from PIL import Image

    short = build_trajectory([_frame(0, (0.5, 0.1, 0.0, 0, 0, 0, 1.0))])  # 1 step
    longer = build_trajectory(
        [_frame(i, (0.2, -0.1, 0.3, 0, 0, 0, 0.0)) for i in range(4)]  # 4 steps
    )
    out = tmp_path / "race.gif"
    render_trajectory_race_gif([short, longer], out, width=560)

    assert out.is_file()
    with Image.open(out) as img:
        assert img.is_animated
        assert img.n_frames == 4  # advances to the longest trajectory's step count
        assert img.size[0] == 560


def test_render_trajectory_race_rejects_no_usable_series(tmp_path: Path) -> None:
    single = Trajectory(
        model="x", action_space="eef_delta", points=((0.0, 0.0, 0.0),), gripper=(None,)
    )
    with pytest.raises(ValueError, match="at least one trajectory with two points"):
        render_trajectory_race_gif([single], tmp_path / "x.gif")


def test_recorded_trajectory_gifs_exist_and_animate() -> None:
    from PIL import Image

    base = REPO_ROOT / "docs" / "assets" / "trajectory"
    for name in ("openvla_trajectory.gif", "smolvla_trajectory.gif", "trajectory_race.gif"):
        gif = base / name
        assert gif.is_file()
        with Image.open(gif) as img:
            assert img.is_animated
            assert img.n_frames > 5
