from __future__ import annotations

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


def no_gpu_demo(progress: int) -> Image.Image:
    image, draw = frame("No GPU demo", "The dummy adapter proves the install and API path.")
    typed = [
        "$ from vla_zoo import load_model",
        "$ model = load_model('dummy')",
        "$ model.predict(image=None, instruction='hello')",
        "VLAAction(action_space='eef_delta', data=[0,0,0,0,0,0,0])",
    ]
    terminal(draw, (44, 130, 856, 410), typed[: max(1, progress)])
    pill(draw, (44, 430), "pip install -e .", BLUE)
    pill(draw, (226, 430), "pytest friendly", GREEN)
    pill(draw, (410, 430), "no model download", YELLOW)
    return image


def ros2_runtime(progress: int) -> Image.Image:
    image, draw = frame(
        "ROS2 runtime",
        "Camera and instruction become published VLAAction messages.",
    )
    boxes = [
        ((54, 175, 234, 255), "camera", BLUE),
        ((54, 300, 234, 380), "instruction", GREEN),
        ((355, 235, 575, 320), "vla_runtime_node", PURPLE),
        ((690, 235, 850, 320), "/vla/action", CYAN),
    ]
    for box, label, color in boxes:
        rounded(draw, box, PANEL)
        draw.text((box[0] + 22, box[1] + 26), label, fill=color, font=STYLE.body)
    arrow(draw, (234, 215), (355, 262), BLUE)
    arrow(draw, (234, 340), (355, 292), GREEN)
    arrow(draw, (575, 278), (690, 278), CYAN)
    for index in range(progress):
        x = 270 + index * 85
        draw.ellipse((x, 270, x + 18, 288), fill=CYAN)
    pill(draw, (54, 420), "dry_run:=true", GREEN)
    pill(draw, (238, 420), "publishes actions", CYAN)
    return image


def remote_gpu(progress: int) -> Image.Image:
    image, draw = frame("Remote GPU path", "Robot CPU can call a GPU workstation over HTTP.")
    rounded(draw, (54, 185, 282, 330), PANEL)
    rounded(draw, (618, 185, 846, 330), PANEL)
    draw.text((92, 220), "robot CPU", fill=TEXT, font=STYLE.body)
    draw.text((82, 260), "ROS2 node", fill=MUTED, font=STYLE.small)
    draw.text((650, 220), "GPU server", fill=TEXT, font=STYLE.body)
    draw.text((646, 260), "vla-zoo serve", fill=MUTED, font=STYLE.small)
    arrow(draw, (282, 250), (618, 250), BLUE)
    arrow(draw, (618, 295), (282, 295), GREEN)
    for index in range(progress):
        x = 332 + index * 54
        draw.ellipse((x, 241, x + 16, 257), fill=BLUE)
    pill(draw, (54, 420), "runtime='remote'", BLUE)
    pill(draw, (284, 420), "same predict() API", GREEN)
    return image


def benchmark(progress: int) -> Image.Image:
    image, draw = frame(
        "Smoke benchmark",
        "The benchmark runner uses the same model adapter boundary.",
    )
    terminal(
        draw,
        (44, 130, 500, 410),
        [
            "$ vla-zoo bench --model dummy",
            "{",
            "  success_rate: 1.0,",
            "  exception_count: 0",
            "}",
        ][: progress + 1],
    )
    rounded(draw, (560, 154, 830, 380), PANEL)
    labels = [
        ("success", GREEN, min(1.0, progress / 4)),
        ("latency", BLUE, 0.75),
        ("stable API", CYAN, 1.0),
    ]
    y = 190
    for label, color, value in labels:
        draw.text((590, y - 4), label, fill=TEXT, font=STYLE.small)
        draw.rounded_rectangle((590, y + 28, 790, y + 46), radius=9, fill=(50, 58, 72))
        draw.rounded_rectangle((590, y + 28, 590 + int(200 * value), y + 46), radius=9, fill=color)
        y += 62
    return image


def adapter_hub(progress: int) -> Image.Image:
    image, draw = frame("Adapter hub", "Built-ins today, external entry points tomorrow.")
    cards = [
        ("dummy", "available", GREEN),
        ("openvla", "optional", BLUE),
        ("pi0", "experimental", YELLOW),
        ("smolvla", "experimental", YELLOW),
        ("groot", "experimental", YELLOW),
    ]
    for index, (name, status, color) in enumerate(cards[: progress + 1]):
        row = index // 2
        col = index % 2
        x = 68 + col * 398
        y = 140 + row * 98
        rounded(draw, (x, y, x + 340, y + 72), PANEL)
        draw.text((x + 22, y + 18), name, fill=TEXT, font=STYLE.body)
        pill(draw, (x + 176, y + 19), status, color)
    draw.text((68, 432), "[project.entry-points.'vla_zoo.adapters']", fill=MUTED, font=STYLE.mono)
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
    save_gif("readme_no_gpu_demo.gif", no_gpu_demo, 4)
    save_gif("readme_ros2_runtime.gif", ros2_runtime, 5)
    save_gif("readme_remote_gpu.gif", remote_gpu, 6)
    save_gif("readme_benchmark.gif", benchmark, 5)
    save_gif("readme_adapter_hub.gif", adapter_hub, 5)


if __name__ == "__main__":
    main()
