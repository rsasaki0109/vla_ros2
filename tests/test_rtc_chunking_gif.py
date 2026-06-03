from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vla_zoo.demo.rtc_chunking_gif import render_rtc_chunking_gif
from vla_zoo.runtime.realtime_chunking import (
    RealtimeChunkingConfig,
    compare_strategies,
    synthetic_chunk_stream,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _report():
    config = RealtimeChunkingConfig(horizon=16, execute_horizon=8, inference_delay_ticks=4)
    chunks = synthetic_chunk_stream(config, chunk_count=6, mode_strength=0.6, seed=7)
    return compare_strategies(chunks, config, source="unit-test")


def test_render_rtc_chunking_gif_writes_animated_gif(tmp_path: Path) -> None:
    from PIL import Image

    report = _report()
    out = tmp_path / "rtc.gif"
    render_rtc_chunking_gif(report, out, width=600)

    assert out.is_file()
    with Image.open(out) as img:
        assert img.format == "GIF"
        assert img.is_animated
        # one frame per revealed tick after the first
        assert img.n_frames == report.naive.emitted.shape[0] - 1
        assert img.size[0] == 600


def test_render_rtc_chunking_gif_needs_two_ticks(tmp_path: Path) -> None:
    report = _report()
    single = replace(
        report,
        naive=replace(report.naive, emitted=np.zeros((1, 3), dtype=np.float32)),
        rtc=replace(report.rtc, emitted=np.zeros((1, 3), dtype=np.float32)),
    )
    with pytest.raises(ValueError, match="at least two control ticks"):
        render_rtc_chunking_gif(single, tmp_path / "x.gif")


def test_recorded_rtc_gif_exists_and_animates() -> None:
    from PIL import Image

    gif = REPO_ROOT / "docs" / "assets" / "rtc_sim" / "rtc_scheduler_sim.gif"
    assert gif.is_file()
    with Image.open(gif) as img:
        assert img.is_animated
        assert img.n_frames > 5
