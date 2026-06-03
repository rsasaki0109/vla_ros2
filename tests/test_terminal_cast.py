from __future__ import annotations

from pathlib import Path

import pytest

from vla_zoo.demo.terminal_cast import (
    CastFrame,
    CastLine,
    build_quickstart_cast,
    render_cast_gif,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_quickstart_cast_reveals_command_then_output() -> None:
    frames = build_quickstart_cast()
    assert len(frames) > 5
    # frames only grow (a terminal never un-prints): line count is non-decreasing
    counts = [len(f.lines) for f in frames]
    assert counts == sorted(counts)
    # the final frame carries the success line and the report path
    final_text = "\n".join(line.text for line in frames[-1].lines)
    assert "runtime boundary works" in final_text
    assert "report.html" in final_text


def test_render_cast_gif_writes_animated_gif(tmp_path: Path) -> None:
    from PIL import Image

    frames = (
        CastFrame((CastLine("$ vla-zoo quickstart", "command"),), 80),
        CastFrame(
            (CastLine("$ vla-zoo quickstart", "command"), CastLine("✓ ok", "ok")),
            120,
        ),
    )
    out = tmp_path / "demo.gif"
    render_cast_gif(frames, out, width=400)

    assert out.is_file()
    with Image.open(out) as img:
        assert img.format == "GIF"
        assert img.is_animated
        assert img.n_frames == 2
        assert img.size[0] == 400


def test_render_cast_gif_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty terminal cast"):
        render_cast_gif((), tmp_path / "x.gif")


def test_recorded_quickstart_demo_gif_exists_and_is_animated() -> None:
    from PIL import Image

    gif = REPO_ROOT / "docs" / "assets" / "quickstart" / "quickstart_demo.gif"
    assert gif.is_file()
    with Image.open(gif) as img:
        assert img.format == "GIF"
        assert img.is_animated
        assert img.n_frames > 5
