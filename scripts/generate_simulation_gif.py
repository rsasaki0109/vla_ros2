from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "simulation_pick_place.gif"
SIZE = (960, 540)

BG = (10, 14, 22)
PANEL = (22, 29, 41)
GRID = (39, 49, 65)
TEXT = (236, 240, 247)
MUTED = (148, 160, 178)
BLUE = (96, 165, 250)
GREEN = (74, 222, 128)
YELLOW = (245, 158, 11)
RED = (239, 68, 68)
CYAN = (34, 211, 238)
PURPLE = (167, 139, 250)
STEEL = (148, 163, 184)
FLOOR = (31, 41, 55)


def font(size: int, *, mono: bool = False) -> ImageFont.ImageFont:
    family = "DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / family
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE = font(34)
BODY = font(22)
SMALL = font(17)
MONO = font(17, mono=True)


@dataclass(frozen=True)
class ArmState:
    time_s: float
    wrist: tuple[float, float]
    target: tuple[float, float]
    block: tuple[float, float]
    gripper: float
    attached: bool
    phase: str
    action: tuple[float, float, float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def smoothstep(value: float) -> float:
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def lerp(start: float, end: float, value: float) -> float:
    return start + (end - start) * value


def mix(a: tuple[float, float], b: tuple[float, float], value: float) -> tuple[float, float]:
    return (lerp(a[0], b[0], value), lerp(a[1], b[1], value))


def segment(
    t: float,
    start: float,
    end: float,
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    value = smoothstep((t - start) / (end - start))
    return mix(a, b, value), value


def ik(
    wrist: tuple[float, float],
    *,
    base: tuple[float, float] = (238.0, 374.0),
    upper: float = 170.0,
    lower: float = 150.0,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    bx, by = base
    wx, wy = wrist
    dx = wx - bx
    dy = wy - by
    distance = clamp(math.hypot(dx, dy), 50.0, upper + lower - 4.0)
    angle = math.atan2(dy, dx)
    cos_elbow = (upper * upper + distance * distance - lower * lower) / (2.0 * upper * distance)
    elbow_angle = angle - math.acos(clamp(cos_elbow, -1.0, 1.0))
    elbow = (bx + upper * math.cos(elbow_angle), by + upper * math.sin(elbow_angle))
    return base, elbow, wrist


def state_at(frame: int, frames: int) -> ArmState:
    t = frame / max(1, frames - 1)
    time_s = t * 6.0
    home = (380.0, 220.0)
    pre_grasp = (610.0, 218.0)
    grasp = (610.0, 315.0)
    lift = (610.0, 198.0)
    pre_place = (748.0, 205.0)
    place = (748.0, 315.0)
    retreat = (510.0, 196.0)
    block_start = (646.0, 352.0)
    block_place = (784.0, 352.0)

    wrist = home
    phase = "observe"
    gripper = 1.0
    attached = False
    target = block_start
    block = block_start

    if t < 0.18:
        wrist, _ = segment(t, 0.0, 0.18, home, pre_grasp)
        phase = "approach"
    elif t < 0.32:
        wrist, _ = segment(t, 0.18, 0.32, pre_grasp, grasp)
        phase = "descend"
    elif t < 0.42:
        wrist = grasp
        gripper = 1.0 - smoothstep((t - 0.32) / 0.10)
        phase = "close gripper"
    elif t < 0.56:
        wrist, _ = segment(t, 0.42, 0.56, grasp, lift)
        gripper = 0.0
        attached = True
        phase = "lift"
    elif t < 0.72:
        wrist, _ = segment(t, 0.56, 0.72, lift, pre_place)
        gripper = 0.0
        attached = True
        phase = "transport"
    elif t < 0.84:
        wrist, _ = segment(t, 0.72, 0.84, pre_place, place)
        gripper = 0.0
        attached = True
        phase = "place"
    elif t < 0.91:
        wrist = place
        gripper = smoothstep((t - 0.84) / 0.07)
        phase = "open gripper"
        block = block_place
    else:
        wrist, _ = segment(t, 0.91, 1.0, place, retreat)
        phase = "retreat"
        block = block_place

    if attached:
        block = (wrist[0] + 37.0, wrist[1] + 37.0)
    target = block_start if t < 0.62 else block_place

    next_t = min(1.0, t + 1.0 / frames)
    next_wrist = wrist
    if next_t < 0.18:
        next_wrist, _ = segment(next_t, 0.0, 0.18, home, pre_grasp)
    elif next_t < 0.32:
        next_wrist, _ = segment(next_t, 0.18, 0.32, pre_grasp, grasp)
    elif next_t < 0.42:
        next_wrist = grasp
    elif next_t < 0.56:
        next_wrist, _ = segment(next_t, 0.42, 0.56, grasp, lift)
    elif next_t < 0.72:
        next_wrist, _ = segment(next_t, 0.56, 0.72, lift, pre_place)
    elif next_t < 0.84:
        next_wrist, _ = segment(next_t, 0.72, 0.84, pre_place, place)
    elif next_t < 0.91:
        next_wrist = place
    else:
        next_wrist, _ = segment(next_t, 0.91, 1.0, place, retreat)

    action = (
        (next_wrist[0] - wrist[0]) / 80.0,
        (next_wrist[1] - wrist[1]) / 80.0,
        0.0,
        1.0 - gripper,
    )
    return ArmState(time_s, wrist, target, block, gripper, attached, phase, action)


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill)


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    font_obj: ImageFont.ImageFont,
) -> None:
    draw.text(xy, text, fill=color, font=font_obj)


def draw_scene(draw: ImageDraw.ImageDraw) -> None:
    draw.rounded_rectangle((34, 104, 926, 444), radius=24, fill=(15, 23, 35))
    draw.rectangle((34, 360, 926, 444), fill=FLOOR)
    for x in range(58, 926, 54):
        draw.line((x, 360, x - 44, 444), fill=GRID, width=1)
    for y in range(384, 444, 22):
        draw.line((34, y, 926, y), fill=(44, 55, 72), width=1)
    draw.line((34, 360, 926, 360), fill=(77, 94, 120), width=2)


def draw_block(draw: ImageDraw.ImageDraw, center: tuple[float, float]) -> None:
    x, y = center
    draw.rounded_rectangle((x - 24, y - 24, x + 24, y + 24), radius=5, fill=RED)
    draw.polygon(
        [(x - 24, y - 24), (x - 6, y - 42), (x + 42, y - 42), (x + 24, y - 24)],
        fill=(248, 113, 113),
    )
    draw.polygon(
        [(x + 24, y - 24), (x + 42, y - 42), (x + 42, y + 6), (x + 24, y + 24)],
        fill=(185, 28, 28),
    )


def draw_target(draw: ImageDraw.ImageDraw, target: tuple[float, float]) -> None:
    x, y = target
    draw.rounded_rectangle((x - 32, y - 32, x + 32, y + 32), radius=7, outline=YELLOW, width=3)
    draw.line((x - 42, y, x + 42, y), fill=YELLOW, width=2)
    draw.line((x, y - 42, x, y + 42), fill=YELLOW, width=2)


def draw_camera(draw: ImageDraw.ImageDraw, pulse: float) -> None:
    x, y = 74, 134
    draw.rounded_rectangle((x, y, x + 76, y + 44), radius=9, fill=(45, 55, 72))
    draw.ellipse((x + 46, y + 10, x + 68, y + 32), fill=BLUE)
    cone = [(x + 58, y + 44), (650, 344), (210, 344)]
    color = (26, 85, 136) if pulse < 0.5 else (30, 105, 170)
    draw.polygon(cone, fill=color)
    draw_text(draw, (74, 108), "/camera/image_raw", BLUE, SMALL)


def draw_arm(draw: ImageDraw.ImageDraw, state: ArmState) -> None:
    base, elbow, wrist = ik(state.wrist)
    draw.rounded_rectangle((172, 358, 304, 408), radius=12, fill=(55, 65, 82))
    draw.ellipse((base[0] - 33, base[1] - 33, base[0] + 33, base[1] + 33), fill=PURPLE)
    draw.line((base, elbow), fill=(107, 124, 148), width=30)
    draw.line((elbow, wrist), fill=STEEL, width=25)
    for point in (base, elbow, wrist):
        x, y = point
        draw.ellipse((x - 19, y - 19, x + 19, y + 19), fill=(226, 232, 240))
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(75, 85, 99))
    gap = 28.0 * state.gripper + 8.0
    x, y = wrist
    draw.line((x, y, x + 38, y - gap), fill=TEXT, width=7)
    draw.line((x, y, x + 38, y + gap), fill=TEXT, width=7)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    state: ArmState,
    history: list[tuple[float, float]],
) -> None:
    rounded(draw, (600, 116, 908, 254), PANEL)
    draw_text(draw, (620, 136), "vla_runtime_node", TEXT, BODY)
    draw_text(draw, (620, 168), "instruction: pick up red block", MUTED, SMALL)
    draw_text(draw, (620, 194), f"phase: {state.phase}", GREEN, SMALL)
    draw_text(draw, (620, 220), f"t={state.time_s:0.2f}s  dry_run=true", MUTED, SMALL)

    rounded(draw, (600, 270, 908, 426), PANEL)
    draw_text(draw, (620, 290), "VLAAction eef_delta[7]", TEXT, BODY)
    labels = ("dx", "dy", "dz", "grip")
    for index, (label, value) in enumerate(zip(labels, state.action, strict=True)):
        y = 328 + index * 22
        draw_text(draw, (620, y - 7), label, MUTED, SMALL)
        draw.rounded_rectangle((674, y, 866, y + 11), radius=5, fill=(45, 55, 72))
        center = 770
        width = int(clamp(value, -1.0, 1.0) * 86)
        color = GREEN if label == "grip" else CYAN
        if width >= 0:
            draw.rounded_rectangle((center, y, center + width, y + 11), radius=5, fill=color)
        else:
            draw.rounded_rectangle((center + width, y, center, y + 11), radius=5, fill=color)

    if len(history) > 1:
        draw.line(history, fill=GREEN, width=3)


def render_frame(frame: int, frames: int, history: list[tuple[float, float]]) -> Image.Image:
    state = state_at(frame, frames)
    history.append(state.wrist)

    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (34, 26), "vla_zoo simulation demo", TEXT, TITLE)
    draw_text(
        draw,
        (36, 70),
        "camera + instruction + robot state -> VLAAction -> dry-run robot motion",
        MUTED,
        SMALL,
    )
    draw_scene(draw)
    draw_camera(draw, frame / frames)
    draw_target(draw, state.target)
    draw_block(draw, state.block)
    draw_arm(draw, state)
    draw_panel(draw, state, history)
    draw_text(
        draw,
        (52, 454),
        "simulated 2D arm: IK, gripper attach/release, action trace",
        MUTED,
        SMALL,
    )
    return image


def main() -> None:
    frames = 72
    history: list[tuple[float, float]] = []
    images = [render_frame(frame, frames, history) for frame in range(frames)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        OUT,
        save_all=True,
        append_images=images[1:] + [images[-1]] * 10,
        duration=55,
        loop=0,
        optimize=True,
    )
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
