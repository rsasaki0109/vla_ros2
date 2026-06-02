"""Render an animated terminal-demo GIF with no external tools (PIL only).

The README hero benefits from a short animated demo, but asciinema/agg are not always
installed and add a toolchain dependency. This module renders a terminal "cast" — a
typed command followed by its output, revealed progressively — straight to an animated
GIF using Pillow (already a core dependency). The result is a deterministic, checked-in
asset that any contributor can regenerate with one command.

The bundled cast illustrates ``vla-zoo quickstart``. It is a presentation asset (an
illustrative terminal demo), not an evidence artifact: the numbers shown are
representative baseline figures, consistent with a real local run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Monospace fonts to try, in order, before falling back to Pillow's default bitmap.
CANDIDATE_MONO_FONTS: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Menlo.ttc",
)

#: Terminal palette (RGB), keyed by a small set of semantic colours.
PALETTE: dict[str, tuple[int, int, int]] = {
    "bg": (13, 18, 28),
    "titlebar": (28, 33, 46),
    "fg": (220, 228, 240),
    "prompt": (76, 194, 255),
    "command": (255, 255, 255),
    "header": (76, 194, 255),
    "ok": (47, 191, 113),
    "muted": (140, 155, 180),
}


@dataclass(frozen=True)
class CastLine:
    """One rendered terminal line and its palette colour."""

    text: str
    color: str = "fg"


@dataclass(frozen=True)
class CastFrame:
    """The full terminal contents for one GIF frame, plus how long to show it."""

    lines: tuple[CastLine, ...]
    duration_ms: int


@dataclass
class _Session:
    """Accumulates terminal lines into a growing list of frames."""

    frames: list[CastFrame] = field(default_factory=list)
    lines: list[CastLine] = field(default_factory=list)

    def _snapshot(self, duration_ms: int, *, pending: CastLine | None = None) -> None:
        shown = (*self.lines, pending) if pending is not None else tuple(self.lines)
        self.frames.append(CastFrame(lines=shown, duration_ms=duration_ms))

    def type_command(self, prompt: str, command: str, *, step: int = 3) -> None:
        """Reveal ``command`` a few characters at a time after a ``prompt``."""

        for end in range(0, len(command) + 1, step):
            pending = CastLine(text=f"{prompt}{command[:end]}", color="command")
            self._snapshot(70, pending=pending)
        # commit the fully typed command line
        self.lines.append(CastLine(text=f"{prompt}{command}", color="command"))
        self._snapshot(360)

    def emit(self, line: CastLine, *, duration_ms: int = 170) -> None:
        """Append one output line and snapshot a frame."""

        self.lines.append(line)
        self._snapshot(duration_ms)

    def hold(self, duration_ms: int) -> None:
        self._snapshot(duration_ms)


def build_quickstart_cast() -> tuple[CastFrame, ...]:
    """Build the frames for the bundled ``vla-zoo quickstart`` terminal demo."""

    session = _Session()
    session.type_command("$ ", "pip install vla_zoo")
    session.emit(CastLine("Successfully installed vla_zoo", color="muted"), duration_ms=260)
    session.emit(CastLine("", color="fg"), duration_ms=80)
    session.type_command("$ ", "vla-zoo quickstart")
    session.emit(CastLine("vla_zoo runtime-boundary quickstart", color="header"))
    session.emit(CastLine("model      space         dim     p50 ms   rate Hz", color="muted"))
    session.emit(CastLine("---------- ------------ ---- ---------- ---------", color="muted"))
    session.emit(CastLine("dummy      eef_delta       7       0.01  73000.00", color="fg"))
    session.emit(CastLine("scripted   eef_delta       7       0.04  16800.00", color="fg"))
    session.emit(CastLine("random     eef_delta       7       0.01  41600.00", color="fg"))
    session.emit(CastLine("", color="fg"), duration_ms=120)
    session.emit(CastLine("✓ runtime boundary works", color="ok"), duration_ms=320)
    session.emit(
        CastLine("Open the report:  ./vla_zoo_quickstart/report.html", color="prompt"),
        duration_ms=300,
    )
    session.hold(2600)
    return tuple(session.frames)


def _load_font(size: int) -> object:
    from PIL import ImageFont

    for path in CANDIDATE_MONO_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_cast_gif(
    frames: tuple[CastFrame, ...],
    path: Path,
    *,
    width: int = 720,
    font_size: int = 17,
    line_height: int = 24,
    padding: int = 18,
    title_height: int = 34,
) -> Path:
    """Render cast ``frames`` into an animated GIF at ``path`` (returns the path)."""

    from PIL import Image, ImageDraw

    if not frames:
        msg = "cannot render an empty terminal cast"
        raise ValueError(msg)

    font = _load_font(font_size)
    max_lines = max(len(frame.lines) for frame in frames)
    body_top = title_height + padding
    height = body_top + max_lines * line_height + padding

    images: list[Image.Image] = []
    durations: list[int] = []
    for frame in frames:
        image = Image.new("RGB", (width, height), PALETTE["bg"])
        draw = ImageDraw.Draw(image)
        # title bar with the classic three traffic-light dots
        draw.rectangle((0, 0, width, title_height), fill=PALETTE["titlebar"])
        for index, dot in enumerate(((237, 106, 94), (245, 191, 79), (98, 197, 84))):
            cx = padding + index * 20
            draw.ellipse((cx, 12, cx + 11, 23), fill=dot)
        draw.text(
            (width // 2 - 36, 9), "vla_zoo", fill=PALETTE["muted"], font=font  # type: ignore[arg-type]
        )
        for row, line in enumerate(frame.lines):
            y = body_top + row * line_height
            draw.text(
                (padding, y),
                line.text,
                fill=PALETTE.get(line.color, PALETTE["fg"]),
                font=font,  # type: ignore[arg-type]
            )
        images.append(image)
        durations.append(frame.duration_ms)

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
