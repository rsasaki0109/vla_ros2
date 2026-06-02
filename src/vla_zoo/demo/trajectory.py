"""Animate a recorded VLA action stream into an end-effector trajectory GIF.

A ``vla_actions.jsonl`` log records the per-step action a real adapter produced. For an
``eef_delta`` action the first three dimensions are end-effector position deltas, so
integrating them open-loop traces out the path the *commanded* deltas would follow. This
module turns that path into an animated GIF (top-down XY + side XZ projections, gripper
signal coloured) using only Pillow — no plotting toolchain.

It is a runtime-path visualisation, **not** a task-success or real-end-effector claim:
the deltas are integrated in action units (not metric), open-loop, and there is no
evidence the physical arm followed this path. It shows what the policy *asked for*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from vla_zoo.benchmark.replay import ReplayFrame

#: Reused fonts (monospace first), falling back to Pillow's scalable default.
_CANDIDATE_FONTS: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
)

_PALETTE = {
    "bg": (13, 18, 28),
    "panel": (22, 29, 43),
    "grid": (40, 52, 73),
    "fg": (220, 228, 240),
    "muted": (130, 145, 170),
    "path": (76, 194, 255),
    "open": (76, 194, 255),
    "closed": (245, 158, 66),
}

TRAJECTORY_NOTE = (
    "Integrated eef_delta action stream (open-loop, action units, not metric). Runtime-path "
    "visualisation of what the policy commanded — NOT a task-success or real-end-effector claim."
)


@dataclass(frozen=True)
class Trajectory:
    """An integrated end-effector path plus per-step gripper signal."""

    model: str
    action_space: str
    points: tuple[tuple[float, float, float], ...]
    gripper: tuple[float | None, ...]

    @property
    def step_count(self) -> int:
        return max(0, len(self.points) - 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "action_space": self.action_space,
            "step_count": self.step_count,
            "points": [list(p) for p in self.points],
            "gripper": list(self.gripper),
        }


def build_trajectory(frames: Sequence[ReplayFrame], *, scale: float = 1.0) -> Trajectory:
    """Integrate the first three action dims into a position path (starting at the origin).

    The 7th dimension (if present) is kept as the gripper signal. Raises on an empty log.
    """

    if not frames:
        msg = "cannot build a trajectory from an empty action log"
        raise ValueError(msg)

    px = py = pz = 0.0
    points: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    gripper: list[float | None] = [None]
    for frame in frames:
        action = frame.action
        dx = action[0] if len(action) > 0 else 0.0
        dy = action[1] if len(action) > 1 else 0.0
        dz = action[2] if len(action) > 2 else 0.0
        px += dx * scale
        py += dy * scale
        pz += dz * scale
        points.append((px, py, pz))
        gripper.append(action[6] if len(action) >= 7 else None)

    return Trajectory(
        model=frames[0].model_name,
        action_space=frames[0].action_space,
        points=tuple(points),
        gripper=tuple(gripper),
    )


def _load_font(size: int) -> object:
    from PIL import ImageFont

    for path in _CANDIDATE_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _grip_color(value: float | None) -> tuple[int, int, int]:
    if value is None:
        return _PALETTE["path"]
    return _PALETTE["open"] if value >= 0.5 else _PALETTE["closed"]


@dataclass(frozen=True)
class _Projection:
    """Maps world coordinates into one panel's pixel space (shared scale)."""

    x0: int
    y0: int
    inner: int
    cx: float
    cy: float
    scale: float

    def to_px(self, wx: float, wy: float) -> tuple[float, float]:
        px = self.x0 + self.inner / 2 + (wx - self.cx) * self.scale
        py = self.y0 + self.inner / 2 - (wy - self.cy) * self.scale  # invert: up is +
        return px, py


def render_trajectory_gif(
    trajectory: Trajectory,
    path: Path,
    *,
    width: int = 760,
    panel: int = 300,
    margin: int = 26,
    gap: int = 20,
    header: int = 46,
    step_ms: int = 130,
    hold_ms: int = 2400,
) -> Path:
    """Render the trajectory as an animated GIF (XY top view + XZ side view)."""

    from PIL import Image, ImageDraw

    pts = trajectory.points
    if len(pts) < 2:
        msg = "need at least two points to animate a trajectory"
        raise ValueError(msg)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    cx, cy, cz = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-6)
    inner = panel - 2 * margin
    scale = inner / (span * 1.15)  # 1.15 leaves a margin around the extremes

    panel_top = header + 8
    xy_x0 = gap + margin
    xz_x0 = gap + panel + gap + margin
    xy = _Projection(xy_x0, panel_top + margin, inner, cx, cy, scale)
    xz = _Projection(xz_x0, panel_top + margin, inner, cx, cz, scale)
    height = panel_top + panel + 30

    font = _load_font(15)
    small = _load_font(12)

    def _panel_box(x0: int) -> tuple[int, int, int, int]:
        return (x0, panel_top, x0 + panel, panel_top + panel)

    def _draw_static(draw: ImageDraw.ImageDraw) -> None:
        for x0, (ha, va) in ((gap, ("X", "Y")), (gap + panel + gap, ("X", "Z"))):
            box = _panel_box(x0)
            draw.rectangle(box, fill=_PALETTE["panel"], outline=_PALETTE["grid"])
            mid_x = x0 + panel / 2
            mid_y = panel_top + panel / 2
            draw.line((x0 + margin, mid_y, x0 + panel - margin, mid_y), fill=_PALETTE["grid"])
            draw.line((mid_x, panel_top + margin, mid_x, panel_top + panel - margin),
                      fill=_PALETTE["grid"])
            draw.text((x0 + panel - 18, mid_y - 16), ha, fill=_PALETTE["muted"], font=small)  # type: ignore[arg-type]
            draw.text((mid_x + 6, panel_top + margin - 4), va, fill=_PALETTE["muted"], font=small)  # type: ignore[arg-type]

    images: list[Image.Image] = []
    durations: list[int] = []
    n = len(pts)
    for i in range(1, n):
        image = Image.new("RGB", (width, height), _PALETTE["bg"])
        draw = ImageDraw.Draw(image)
        title = f"{trajectory.model} commanded EEF trajectory  ·  step {i}/{n - 1}"
        draw.text((gap, 14), title, fill=_PALETTE["fg"], font=font)  # type: ignore[arg-type]
        _draw_static(draw)
        for proj, (a, b) in ((xy, (0, 1)), (xz, (0, 2))):
            for k in range(1, i + 1):
                p0, p1 = pts[k - 1], pts[k]
                draw.line(
                    (*proj.to_px(p0[a], p0[b]), *proj.to_px(p1[a], p1[b])),
                    fill=_PALETTE["path"],
                    width=2,
                )
            hx, hy = proj.to_px(pts[i][a], pts[i][b])
            color = _grip_color(trajectory.gripper[i])
            draw.ellipse((hx - 5, hy - 5, hx + 5, hy + 5), fill=color)
        draw.text((gap, height - 18), TRAJECTORY_NOTE[:96], fill=_PALETTE["muted"], font=small)  # type: ignore[arg-type]
        images.append(image)
        durations.append(step_ms)

    durations[-1] = hold_ms  # linger on the completed path

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
