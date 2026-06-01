from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path("docs/assets")
DATA_PATH = ASSETS / "sample_runtime_comparison.json"
DASHBOARD_PREVIEW = ASSETS / "dashboard_preview.png"
SOCIAL_PREVIEW = ASSETS / "social_preview.png"
SIM_GIF = ASSETS / "simulation_pick_place.gif"


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        name = "DejaVuSansMono.ttf"
    elif bold:
        name = "DejaVuSans-Bold.ttf"
    else:
        name = "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size=size)


FONT_14 = font(14)
FONT_16 = font(16)
FONT_18 = font(18)
FONT_22 = font(22, bold=True)
FONT_28 = font(28, bold=True)
FONT_36 = font(36, bold=True)
FONT_48 = font(48, bold=True)
FONT_76 = font(76, bold=True)
MONO_15 = font(15, mono=True)


def load_records() -> list[dict[str, Any]]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    width: int,
    line_height: int,
    fill: str,
    text_font: ImageFont.FreeTypeFont,
) -> int:
    x, y = xy
    for line in textwrap.wrap(text, width=width):
        draw.text((x, y), line, fill=fill, font=text_font)
        y += line_height
    return y


def rounded_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = "#ffffff",
    outline: str = "#d7dee8",
) -> None:
    draw.rounded_rectangle(box, radius=16, fill=fill, outline=outline, width=1)


def metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
) -> None:
    rounded_panel(draw, box)
    x0, y0, _, _ = box
    draw.text((x0 + 18, y0 + 16), label, fill="#5b687a", font=FONT_16)
    draw.text((x0 + 18, y0 + 45), value, fill="#172033", font=FONT_36)


def bar(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: float,
    maximum: float,
    color: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=7, fill="#e5e7eb")
    if maximum <= 0:
        return
    width = max(2, int((x1 - x0) * min(value / maximum, 1.0)))
    draw.rounded_rectangle((x0, y0, x0 + width, y1), radius=7, fill=color)


def generate_dashboard_preview(records: list[dict[str, Any]]) -> None:
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), "#f4f7fb")
    draw = ImageDraw.Draw(image)

    draw.text((54, 42), "vla_zoo Runtime Dashboard", fill="#172033", font=FONT_48)
    draw_text_block(
        draw,
        (56, 106),
        "Operational view for adapter readiness, latency budget, query health, and triage.",
        width=78,
        line_height=25,
        fill="#5b687a",
        text_font=FONT_18,
    )

    ok_count = sum(1 for item in records if item.get("ok"))
    total_queries = sum(int(item.get("adapter_queries") or 0) for item in records)
    total_errors = sum(int(item.get("adapter_errors") or 0) for item in records)
    latencies = [
        float(item["mean_latency_ms"])
        for item in records
        if item.get("mean_latency_ms") is not None
    ]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    health = round(ok_count / max(len(records), 1) * 100)
    frames = sum(int(item.get("frames") or 0) for item in records)

    cards = [
        ("health", f"{health}%"),
        ("ready", f"{ok_count}/{len(records)}"),
        ("queries", str(total_queries)),
        ("error rate", f"{(total_errors / max(total_queries, 1) * 100):.1f}%"),
        ("avg latency", f"{avg_latency:.2f} ms"),
        ("frames", str(frames)),
    ]
    card_w, card_h = 203, 112
    for index, (label, value) in enumerate(cards):
        x = 54 + index * (card_w + 18)
        metric_card(draw, (x, 176, x + card_w, 176 + card_h), label, value)

    rounded_panel(draw, (54, 322, 670, 578))
    draw.text((76, 344), "Fleet Health", fill="#172033", font=FONT_28)
    draw.rounded_rectangle((78, 394, 248, 530), radius=8, fill="#f8fafc", outline="#d7dee8")
    draw.text((112, 422), f"{health}%", fill="#172033", font=FONT_48)
    draw.text((112, 482), "runtime health", fill="#64748b", font=FONT_14)
    states = [
        ("ready", ok_count, "#16a34a"),
        ("failed", len(records) - ok_count, "#e11d48"),
        ("errors", total_errors, "#ca8a04"),
    ]
    y = 404
    max_state = max(1, *(count for _, count, _ in states))
    for label, value, color in states:
        draw.text((276, y - 4), label, fill="#334155", font=FONT_16)
        bar(draw, (390, y, 590, y + 14), value, max_state, color)
        draw.text((608, y - 4), str(value), fill="#172033", font=FONT_14)
        y += 42

    rounded_panel(draw, (704, 322, 1346, 578))
    draw.text((726, 344), "Triage Queue", fill="#172033", font=FONT_28)
    y = 394
    for item in records:
        if item.get("ok"):
            continue
        name = str(item.get("model_name", "unknown"))
        err = str(item.get("last_error", "adapter failed"))
        draw.rounded_rectangle((728, y, 1322, y + 56), radius=8, fill="#fff7ed", outline="#fed7aa")
        draw.text((746, y + 10), name, fill="#075985", font=MONO_15)
        draw.text((902, y + 10), "failed", fill="#9f1239", font=FONT_14)
        draw_text_block(
            draw,
            (746, y + 30),
            err,
            width=72,
            line_height=16,
            fill="#334155",
            text_font=FONT_14,
        )
        y += 64
        if y > 538:
            break

    rounded_panel(draw, (54, 610, 670, 818))
    draw.text((76, 632), "Latency Budget", fill="#172033", font=FONT_28)
    max_latency = max(latencies, default=1.0)
    y = 686
    for item in records:
        name = str(item.get("model_name", "unknown"))
        value = float(item.get("mean_latency_ms") or 0.0)
        draw.text((76, y - 3), name, fill="#075985", font=MONO_15)
        bar(draw, (220, y, 588, y + 14), value, max_latency, "#16a34a")
        label = f"{value:.2f}" if item.get("mean_latency_ms") is not None else "-"
        draw.text((604, y - 4), label, fill="#172033", font=FONT_14)
        y += 28

    rounded_panel(draw, (704, 610, 1346, 818))
    draw.text((726, 632), "Records", fill="#172033", font=FONT_28)
    headers = ("model", "runtime", "health", "errors")
    xs = (726, 910, 1042, 1194)
    for x, header in zip(xs, headers, strict=True):
        draw.text((x, 682), header.upper(), fill="#64748b", font=FONT_14)
    y = 720
    for item in records:
        ok = bool(item.get("ok"))
        score = "100%" if ok else "0%"
        color = "#bbf7d0" if ok else "#fecdd3"
        text_color = "#14532d" if ok else "#881337"
        draw.text((726, y), str(item.get("model_name", "unknown")), fill="#075985", font=MONO_15)
        draw.text((910, y), str(item.get("runtime", "-")), fill="#172033", font=FONT_14)
        draw.rounded_rectangle((1042, y - 3, 1116, y + 21), radius=12, fill=color)
        draw.text((1056, y), score, fill=text_color, font=FONT_14)
        draw.text((1194, y), str(item.get("adapter_errors", 0)), fill="#172033", font=FONT_14)
        y += 28

    rounded_panel(draw, (54, 842, 1346, 884), fill="#0f172a", outline="#1f2937")
    command = (
        "vla-zoo compare dashboard --results results/vla_runtime_comparison.json "
        "  --out results/vla_runtime_dashboard.html"
    )
    draw.text((78, 854), command, fill="#e5e7eb", font=MONO_15)

    image.save(DASHBOARD_PREVIEW, optimize=True)


def generate_social_preview(records: list[dict[str, Any]]) -> None:
    width, height = 1200, 630
    image = Image.new("RGB", (width, height), "#0f172a")
    draw = ImageDraw.Draw(image, "RGBA")

    if SIM_GIF.exists():
        with Image.open(SIM_GIF) as gif:
            frame = gif.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        image.paste(frame)
        draw.rectangle((0, 0, width, height), fill=(3, 7, 18, 166))

    draw.rounded_rectangle((54, 52, 1146, 578), radius=28, fill=(248, 250, 252, 235))
    draw.text((92, 92), "vla_zoo", fill="#0f172a", font=FONT_76)
    draw_text_block(
        draw,
        (96, 186),
        "ROS2-native runtime, benchmark, and adapter hub for Vision-Language-Action models.",
        width=32,
        line_height=34,
        fill="#334155",
        text_font=FONT_22,
    )

    ok_count = sum(1 for item in records if item.get("ok"))
    total_queries = sum(int(item.get("adapter_queries") or 0) for item in records)
    stats = [
        ("models", str(len(records))),
        ("ok", str(ok_count)),
        ("queries", str(total_queries)),
    ]
    for index, (label, value) in enumerate(stats):
        x = 96 + index * 176
        draw.rounded_rectangle((x, 374, x + 146, 480), radius=18, fill="#ffffff", outline="#d7dee8")
        draw.text((x + 18, 392), label, fill="#64748b", font=FONT_16)
        draw.text((x + 18, 420), value, fill="#0f172a", font=FONT_36)

    draw.rounded_rectangle((690, 110, 1098, 498), radius=18, fill="#ffffff", outline="#d7dee8")
    draw.text((718, 140), "Runtime paths", fill="#172033", font=FONT_28)
    paths = ["Python API", "ROS2 node", "Remote server", "PyBullet compare"]
    y = 202
    for item in paths:
        draw.rounded_rectangle((722, y, 760, y + 38), radius=19, fill="#e0f2fe")
        draw.ellipse((735, y + 13, 747, y + 25), fill="#0369a1")
        draw.text((778, y + 5), item, fill="#172033", font=FONT_22)
        y += 62

    draw.text((96, 520), "rsasaki0109.github.io/vla_zoo", fill="#0369a1", font=FONT_22)
    image.save(SOCIAL_PREVIEW, optimize=True)


def main() -> None:
    records = load_records()
    generate_dashboard_preview(records)
    generate_social_preview(records)
    print(DASHBOARD_PREVIEW)
    print(SOCIAL_PREVIEW)


if __name__ == "__main__":
    main()
