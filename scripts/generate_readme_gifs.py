from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
SIZE = (900, 500)
BG = (12, 16, 22)
PANEL = (24, 31, 42)
TEXT = (233, 238, 246)
MUTED = (143, 154, 170)
GREEN = (72, 187, 120)
BLUE = (96, 165, 250)
YELLOW = (245, 158, 11)
RED = (248, 113, 113)
PURPLE = (167, 139, 250)
CYAN = (34, 211, 238)
STEEL = (124, 139, 161)
FLOOR = (31, 41, 55)
CUBE = (239, 68, 68)


@dataclass(frozen=True)
class Style:
    title: ImageFont.ImageFont
    body: ImageFont.ImageFont
    mono: ImageFont.ImageFont
    small: ImageFont.ImageFont


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


STYLE = Style(title=font(34), body=font(24), mono=font(22), small=font(18))


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill)


def pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), label, font=STYLE.small)
    width = bbox[2] - bbox[0] + 28
    draw.rounded_rectangle((x, y, x + width, y + 34), radius=17, fill=color)
    draw.text((x + 14, y + 7), label, fill=(8, 12, 18), font=STYLE.small)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 5,
) -> None:
    draw.line((start, end), fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if x2 >= x1:
        points = [(x2, y2), (x2 - 18, y2 - 10), (x2 - 18, y2 + 10)]
    else:
        points = [(x2, y2), (x2 + 18, y2 - 10), (x2 + 18, y2 + 10)]
    draw.polygon(points, fill=color)


def terminal(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str]) -> None:
    rounded(draw, box, PANEL)
    x1, y1, x2, _ = box
    draw.rounded_rectangle((x1, y1, x2, y1 + 42), radius=18, fill=(34, 42, 56))
    for index, color in enumerate((RED, YELLOW, GREEN)):
        draw.ellipse((x1 + 18 + index * 24, y1 + 14, x1 + 30 + index * 24, y1 + 26), fill=color)
    y = y1 + 64
    for line in lines:
        draw.text((x1 + 24, y), line, fill=TEXT if line.startswith("$") else GREEN, font=STYLE.mono)
        y += 34


def frame(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)
    draw.text((42, 32), title, fill=TEXT, font=STYLE.title)
    draw.text((44, 78), subtitle, fill=MUTED, font=STYLE.small)
    return image, draw


def lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * max(0.0, min(1.0, t))


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ik_arm(
    base: tuple[float, float],
    wrist: tuple[float, float],
    *,
    upper: float = 150.0,
    lower: float = 135.0,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    bx, by = base
    wx, wy = wrist
    dx = wx - bx
    dy = wy - by
    distance = max(30.0, min(math.hypot(dx, dy), upper + lower - 4.0))
    angle = math.atan2(dy, dx)
    cos_elbow = (upper * upper + distance * distance - lower * lower) / (2.0 * upper * distance)
    elbow_angle = angle - math.acos(max(-1.0, min(1.0, cos_elbow)))
    elbow = (bx + upper * math.cos(elbow_angle), by + upper * math.sin(elbow_angle))
    return base, elbow, wrist


def draw_workspace(draw: ImageDraw.ImageDraw) -> None:
    draw.rounded_rectangle((42, 118, 858, 418), radius=22, fill=(17, 24, 34))
    draw.rectangle((42, 340, 858, 418), fill=FLOOR)
    for x in range(70, 850, 70):
        draw.line((x, 342, x - 32, 418), fill=(42, 52, 67), width=1)
    draw.line((42, 340, 858, 340), fill=(73, 89, 111), width=2)


def draw_camera(draw: ImageDraw.ImageDraw, origin: tuple[int, int], pulse: float) -> None:
    x, y = origin
    draw.rounded_rectangle((x, y, x + 74, y + 42), radius=10, fill=(45, 55, 72))
    draw.ellipse((x + 45, y + 9, x + 67, y + 31), fill=BLUE)
    cone = [
        (x + 58, y + 42),
        (x + 190, y + 216),
        (x + 28, y + 216),
    ]
    color = (33, 121, 190) if pulse < 0.5 else (36, 150, 220)
    draw.polygon(cone, fill=color)
    draw.text((x, y - 28), "camera", fill=BLUE, font=STYLE.small)


def draw_block(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    label: str = "red block",
) -> None:
    x, y = center
    draw.rounded_rectangle((x - 28, y - 28, x + 28, y + 28), radius=6, fill=CUBE)
    draw.polygon(
        [(x - 28, y - 28), (x - 8, y - 46), (x + 48, y - 46), (x + 28, y - 28)],
        fill=(248, 113, 113),
    )
    draw.polygon(
        [(x + 28, y - 28), (x + 48, y - 46), (x + 48, y + 10), (x + 28, y + 28)],
        fill=(185, 28, 28),
    )
    draw.text((int(x - 42), int(y + 36)), label, fill=MUTED, font=STYLE.small)


def draw_robot_arm(
    draw: ImageDraw.ImageDraw,
    wrist: tuple[float, float],
    *,
    gripper_closed: bool,
    base: tuple[float, float] = (235.0, 337.0),
) -> None:
    base_pt, elbow, hand = ik_arm(base, wrist)
    draw.rounded_rectangle(
        (base[0] - 66, base[1] - 16, base[0] + 66, base[1] + 36),
        radius=12,
        fill=(55, 65, 82),
    )
    draw.ellipse((base[0] - 31, base[1] - 31, base[0] + 31, base[1] + 31), fill=PURPLE)
    draw.line((base_pt, elbow), fill=STEEL, width=28)
    draw.line((elbow, hand), fill=(148, 163, 184), width=24)
    for joint in (base_pt, elbow, hand):
        x, y = joint
        draw.ellipse((x - 18, y - 18, x + 18, y + 18), fill=(226, 232, 240))
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(75, 85, 99))
    hx, hy = hand
    gap = 11 if gripper_closed else 25
    draw.line((hx, hy, hx + 36, hy - gap), fill=TEXT, width=7)
    draw.line((hx, hy, hx + 36, hy + gap), fill=TEXT, width=7)


def draw_action_bars(draw: ImageDraw.ImageDraw, xy: tuple[int, int], pulse: float) -> None:
    x, y = xy
    rounded(draw, (x, y, x + 230, y + 124), PANEL)
    draw.text((x + 18, y + 16), "VLAAction", fill=TEXT, font=STYLE.body)
    for index, name in enumerate(("dx", "dy", "dz")):
        row_y = y + 56 + index * 22
        draw.text((x + 18, row_y - 8), name, fill=MUTED, font=STYLE.small)
        width = int(46 + 80 * abs(math.sin(pulse + index)))
        draw.rounded_rectangle((x + 64, row_y, x + 64 + width, row_y + 10), radius=5, fill=CYAN)


def robot_motion(
    progress: int,
    frame_count: int,
) -> tuple[tuple[float, float], tuple[float, float], bool]:
    phase = progress / max(1, frame_count - 1)
    reach = ease(min(phase / 0.45, 1.0))
    closed = phase > 0.48
    lift = ease((phase - 0.55) / 0.35)
    block_start = (590.0, 312.0)
    wrist_reach = (block_start[0] - 36.0, block_start[1] - 12.0)
    wrist_lift = (block_start[0] - 36.0, 216.0)
    wrist = (
        lerp(355.0, wrist_reach[0], reach),
        lerp(235.0, lerp(wrist_reach[1], wrist_lift[1], lift), reach),
    )
    block = block_start if not closed else (wrist[0] + 58.0, wrist[1] + 12.0)
    return wrist, block, closed


def no_gpu_demo(progress: int) -> Image.Image:
    frame_count = 9
    image, draw = frame("No GPU demo", "A robot-shaped smoke path with the dummy adapter.")
    draw_workspace(draw)
    wrist, block, closed = robot_motion(progress, frame_count)
    draw_camera(draw, (84, 148), progress / frame_count)
    draw_block(draw, block)
    draw_robot_arm(draw, wrist, gripper_closed=closed)
    draw_action_bars(draw, (630, 156), progress * 0.8)
    typed = [
        "$ load_model('dummy')",
        "$ predict('pick up red block')",
        "VLAAction: eef_delta[7]",
    ]
    line_start = max(0, min(progress // 3, 2))
    terminal(draw, (44, 424, 856, 492), typed[line_start : min(3, line_start + 2)])
    pill(draw, (42, 92), "no GPU", GREEN)
    pill(draw, (170, 92), "no model download", YELLOW)
    return image


def ros2_runtime(progress: int) -> Image.Image:
    frame_count = 9
    image, draw = frame(
        "ROS2 runtime",
        "A dry-run node publishes actions while the robot visualization moves.",
    )
    draw_workspace(draw)
    wrist, block, closed = robot_motion(progress, frame_count)
    draw_block(draw, block)
    draw_robot_arm(draw, wrist, gripper_closed=closed)
    rounded(draw, (54, 132, 252, 208), PANEL)
    rounded(draw, (350, 132, 570, 208), PANEL)
    rounded(draw, (662, 132, 842, 208), PANEL)
    draw.text((78, 154), "/camera", fill=BLUE, font=STYLE.body)
    draw.text((370, 154), "vla_runtime_node", fill=PURPLE, font=STYLE.small)
    draw.text((690, 154), "/vla/action", fill=CYAN, font=STYLE.small)
    arrow(draw, (252, 170), (350, 170), BLUE, width=4)
    arrow(draw, (570, 170), (662, 170), CYAN, width=4)
    packet_x = 270 + (progress % frame_count) * 42
    draw.ellipse((packet_x, 161, packet_x + 18, 179), fill=CYAN)
    pill(draw, (54, 420), "dry_run:=true", GREEN)
    pill(draw, (238, 420), "publishes actions", CYAN)
    return image


def remote_gpu(progress: int) -> Image.Image:
    frame_count = 10
    image, draw = frame("Remote GPU path", "Robot CPU streams observations to a GPU box.")
    draw_workspace(draw)
    wrist, block, closed = robot_motion(progress, frame_count)
    draw_block(draw, block)
    draw_robot_arm(draw, wrist, gripper_closed=closed, base=(188.0, 337.0))
    rounded(draw, (430, 128, 622, 210), PANEL)
    rounded(draw, (678, 128, 850, 210), PANEL)
    draw.text((452, 154), "robot CPU", fill=TEXT, font=STYLE.body)
    draw.text((700, 154), "GPU server", fill=TEXT, font=STYLE.body)
    arrow(draw, (622, 154), (678, 154), BLUE, width=4)
    arrow(draw, (678, 188), (622, 188), GREEN, width=4)
    obs_x = 630 + (progress % frame_count) * 5
    act_x = 662 - (progress % frame_count) * 5
    draw.ellipse((obs_x, 146, obs_x + 14, 160), fill=BLUE)
    draw.ellipse((act_x, 181, act_x + 14, 195), fill=GREEN)
    pill(draw, (54, 420), "runtime='remote'", BLUE)
    pill(draw, (284, 420), "same predict() API", GREEN)
    return image


def benchmark(progress: int) -> Image.Image:
    frame_count = 8
    image, draw = frame(
        "Smoke benchmark",
        "The same adapter contract drives repeatable motion and metrics.",
    )
    draw_workspace(draw)
    wrist, block, closed = robot_motion(progress, frame_count)
    draw_block(draw, block)
    draw_robot_arm(draw, wrist, gripper_closed=closed, base=(188.0, 337.0))
    terminal(
        draw,
        (500, 128, 858, 292),
        [
            "$ vla-zoo bench --model dummy",
            "{",
            "  success_rate: 1.0,",
            "  exception_count: 0",
            "}",
        ][: progress + 1],
    )
    rounded(draw, (500, 312, 858, 408), PANEL)
    labels = [
        ("success", GREEN, min(1.0, progress / 4)),
        ("latency", BLUE, 0.75),
    ]
    y = 336
    for label, color, value in labels:
        draw.text((520, y - 4), label, fill=TEXT, font=STYLE.small)
        draw.rounded_rectangle((622, y + 2, 820, y + 16), radius=7, fill=(50, 58, 72))
        draw.rounded_rectangle((622, y + 2, 622 + int(198 * value), y + 16), radius=7, fill=color)
        y += 38
    return image


def adapter_hub(progress: int) -> Image.Image:
    image, draw = frame("Adapter hub", "Swap adapters while the robot-facing API stays stable.")
    draw_workspace(draw)
    wrist, block, closed = robot_motion(progress, 8)
    draw_block(draw, block)
    draw_robot_arm(draw, wrist, gripper_closed=closed, base=(180.0, 337.0))
    cards = [
        ("dummy", "available", GREEN),
        ("openvla", "optional", BLUE),
        ("pi0", "experimental", YELLOW),
        ("smolvla", "experimental", YELLOW),
        ("groot", "experimental", YELLOW),
    ]
    for index, (name, status, color) in enumerate(cards[: progress + 1]):
        x = 520
        y = 126 + index * 58
        rounded(draw, (x, y, x + 320, y + 44), PANEL)
        draw.text((x + 22, y + 18), name, fill=TEXT, font=STYLE.body)
        pill(draw, (x + 174, y + 6), status, color)
    draw.text((64, 430), "one robot runtime, many VLA adapters", fill=MUTED, font=STYLE.mono)
    return image


def save_gif(name: str, maker: Callable[[int], Image.Image], frame_count: int = 6) -> None:
    frames = [maker(index) for index in range(frame_count)]
    output = OUT / name
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:] + [frames[-1]] * 3,
        duration=520,
        loop=0,
        optimize=True,
    )
    print(output.relative_to(ROOT))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    save_gif("readme_no_gpu_demo.gif", no_gpu_demo, 9)
    save_gif("readme_ros2_runtime.gif", ros2_runtime, 9)
    save_gif("readme_remote_gpu.gif", remote_gpu, 10)
    save_gif("readme_benchmark.gif", benchmark, 8)
    save_gif("readme_adapter_hub.gif", adapter_hub, 8)


if __name__ == "__main__":
    main()
