"""Animate the Real-Time Chunking scheduler simulation into a comparison GIF.

The numbers in ``vla-zoo rtc-sim`` (naive async vs RTC freeze chunk-boundary continuity)
are most convincing when you can *see* them: this module renders the two emitted control
streams as time-series, stacked, with the chunk boundaries marked. The naive panel jumps
at every boundary (switching between independently sampled chunks); the RTC-freeze panel
stays continuous. Pillow only — no plotting toolchain.

It visualises a **runtime scheduling property** (continuity across chunk swaps under
inference latency), not a policy-quality or task-success claim. The chunk stream is the
same deterministic simulation input as ``rtc-sim``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vla_zoo.demo.trajectory import _PALETTE, RACE_COLORS, _load_font
from vla_zoo.runtime.realtime_chunking import RTCSimReport


@dataclass(frozen=True)
class _TimeAxis:
    """Maps (tick, value) into one panel's pixel space (shared scales)."""

    x0: int
    top: int
    inner_w: int
    inner_h: int
    pad: int
    ticks: int
    vmin: float
    vspan: float

    def x_of(self, tick: int) -> float:
        denom = max(1, self.ticks - 1)
        return self.x0 + self.pad + tick / denom * (self.inner_w - 2 * self.pad)

    def y_of(self, value: float) -> float:
        frac = (value - self.vmin) / self.vspan
        return self.top + self.pad + (1.0 - frac) * (self.inner_h - 2 * self.pad)


def render_rtc_chunking_gif(
    report: RTCSimReport,
    path: Path,
    *,
    width: int = 760,
    panel_h: int = 150,
    margin: int = 24,
    gap: int = 16,
    header: int = 70,
    footer: int = 22,
    step_ms: int = 70,
    hold_ms: int = 2600,
) -> Path:
    """Render naive-async vs RTC-freeze emitted control streams as a stacked comparison GIF."""

    from PIL import Image, ImageDraw

    naive = np.asarray(report.naive.emitted, dtype=np.float32)
    rtc = np.asarray(report.rtc.emitted, dtype=np.float32)
    ticks, dims = naive.shape
    if ticks < 2:
        msg = "need at least two control ticks to animate"
        raise ValueError(msg)

    stacked = np.concatenate([naive, rtc], axis=0)
    vmin, vmax = float(stacked.min()), float(stacked.max())
    vspan = max(vmax - vmin, 1e-6)
    inner_w = width - 2 * margin
    pad = 16

    top_naive = header
    top_rtc = header + panel_h + gap
    height = header + 2 * panel_h + gap + footer
    boundaries = report.naive.boundary_indices

    axis_naive = _TimeAxis(margin, top_naive, inner_w, panel_h, pad, ticks, vmin, vspan)
    axis_rtc = _TimeAxis(margin, top_rtc, inner_w, panel_h, pad, ticks, vmin, vspan)

    font = _load_font(15)
    small = _load_font(12)
    colors = [RACE_COLORS[d % len(RACE_COLORS)] for d in range(dims)]

    def _panel(draw: ImageDraw.ImageDraw, axis: _TimeAxis, label: str, jump: float) -> None:
        box = (margin, axis.top, margin + inner_w, axis.top + panel_h)
        draw.rectangle(box, fill=_PALETTE["panel"], outline=_PALETTE["grid"])
        mid_y = axis.y_of((vmin + vmax) / 2)
        draw.line((margin + pad, mid_y, margin + inner_w - pad, mid_y), fill=_PALETTE["grid"])
        for b in boundaries:  # dashed chunk-boundary markers
            bx = axis.x_of(b)
            for yy in range(int(axis.top + pad), int(axis.top + panel_h - pad), 8):
                draw.line((bx, yy, bx, yy + 4), fill=_PALETTE["grid"])
        draw.text((margin + 6, axis.top + 4), label, fill=_PALETTE["fg"], font=small)  # type: ignore[arg-type]
        draw.text(
            (margin + inner_w - 168, axis.top + 4),
            f"mean boundary jump {jump:.3f}",
            fill=_PALETTE["muted"],
            font=small,  # type: ignore[arg-type]
        )

    def _draw_series(
        draw: ImageDraw.ImageDraw, axis: _TimeAxis, data: np.ndarray, upto: int
    ) -> None:
        for d in range(dims):
            for k in range(1, upto + 1):
                draw.line(
                    (axis.x_of(k - 1), axis.y_of(float(data[k - 1, d])),
                     axis.x_of(k), axis.y_of(float(data[k, d]))),
                    fill=colors[d],
                    width=2,
                )
            hx, hy = axis.x_of(upto), axis.y_of(float(data[upto, d]))
            draw.ellipse((hx - 3, hy - 3, hx + 3, hy + 3), fill=colors[d])

    reduction = report.boundary_jump_reduction * 100
    images: list[Image.Image] = []
    durations: list[int] = []
    for i in range(1, ticks):
        image = Image.new("RGB", (width, height), _PALETTE["bg"])
        draw = ImageDraw.Draw(image)
        draw.text(
            (margin, 12),
            f"Real-Time Chunking: naive async vs RTC freeze  ·  tick {i}/{ticks - 1}",
            fill=_PALETTE["fg"],
            font=font,  # type: ignore[arg-type]
        )
        draw.text(
            (margin, 36),
            f"freeze cuts the chunk-boundary jump {reduction:.0f}%  "
            f"(H={report.config.horizon}, s={report.config.execute_horizon}, "
            f"d={report.config.inference_delay_ticks})",
            fill=_PALETTE["muted"],
            font=small,  # type: ignore[arg-type]
        )
        _panel(draw, axis_naive, "naive async", report.naive.mean_boundary_jump)
        _panel(draw, axis_rtc, "RTC freeze", report.rtc.mean_boundary_jump)
        _draw_series(draw, axis_naive, naive, i)
        _draw_series(draw, axis_rtc, rtc, i)
        draw.text(
            (margin, height - 16),
            "Runtime scheduling property under inference latency — not a policy-quality claim.",
            fill=_PALETTE["muted"],
            font=small,  # type: ignore[arg-type]
        )
        images.append(image)
        durations.append(step_ms)

    durations[-1] = hold_ms

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=True,
    )
    return path
