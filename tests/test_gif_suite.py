from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from vla_zoo.cli.main import app
from vla_zoo.demo.gif_suite import (
    PyBulletGifResult,
    PyBulletGifSpec,
    build_pybullet_gif_specs,
    check_gif_suite,
    format_gif_check_markdown,
    format_gif_gallery_markdown,
    format_gif_report_html,
    render_pybullet_gif_suite,
    resolve_pybullet_tasks,
    write_gif_gallery,
    write_gif_manifest,
    write_gif_report_html,
)
from vla_zoo.demo.pybullet import PyBulletDemoConfig


def _write_test_gif(path: Path, *, size: tuple[int, int] = (4, 3), frames: int = 3) -> None:
    images: list[Image.Image] = []
    for index in range(frames):
        image = Image.new("RGB", size)
        pixels = image.load()
        for y in range(size[1]):
            for x in range(size[0]):
                pixels[x, y] = (
                    min(255, index * 40 + x * 30),
                    min(255, 40 + y * 50),
                    max(0, 220 - x * 20 - y * 20),
                )
        images.append(image)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=40,
        loop=0,
    )


def _write_test_manifest(tmp_path: Path, *, gif_name: str = "demo.gif") -> Path:
    gif_path = tmp_path / gif_name
    _write_test_gif(gif_path)
    result = PyBulletGifResult(
        spec=PyBulletGifSpec(
            model_name="dummy",
            task_id="pick_red_block",
            instruction="pick up the red block",
            out=gif_path,
        ),
        ok=True,
        frames=3,
        bytes=gif_path.stat().st_size,
    )
    manifest = tmp_path / "gif_manifest.json"
    write_gif_manifest(manifest, [result])
    return manifest


def test_resolve_pybullet_tasks_all() -> None:
    tasks = resolve_pybullet_tasks("all")

    assert len(tasks) >= 3
    assert {task.task_id for task in tasks} >= {
        "pick_red_block",
        "move_red_block_left",
        "move_red_block_right",
    }


def test_build_pybullet_gif_specs_crosses_models_and_tasks(tmp_path: Path) -> None:
    tasks = resolve_pybullet_tasks("pick_red_block,move_red_block_left")
    specs = build_pybullet_gif_specs(
        models=("dummy", "scripted"),
        tasks=tasks,
        out_dir=tmp_path,
        render_stride=12,
    )

    assert len(specs) == 4
    assert specs[0].out == tmp_path / "simulation_pick_red_block_dummy.gif"
    assert specs[0].render_stride == 12
    assert "--task pick_red_block" in specs[0].command()
    assert specs[-1].out.name == "simulation_move_red_block_left_scripted.gif"


def test_render_pybullet_gif_suite_accepts_fake_renderer(tmp_path: Path) -> None:
    specs = build_pybullet_gif_specs(
        models=("dummy",),
        tasks=resolve_pybullet_tasks("pick_red_block"),
        out_dir=tmp_path,
    )

    def fake_renderer(config: PyBulletDemoConfig) -> dict[str, object]:
        config.out.parent.mkdir(parents=True, exist_ok=True)
        config.out.write_bytes(b"GIF89a")
        return {"frames": 12, "out": str(config.out)}

    results = render_pybullet_gif_suite(specs, renderer=fake_renderer)

    assert len(results) == 1
    assert results[0].ok
    assert results[0].frames == 12
    assert results[0].bytes == 6


def test_gif_gallery_markdown_is_readme_ready(tmp_path: Path) -> None:
    specs = build_pybullet_gif_specs(
        models=("dummy",),
        tasks=resolve_pybullet_tasks("pick_red_block"),
        out_dir=tmp_path,
    )
    results = render_pybullet_gif_suite(
        specs,
        renderer=lambda config: {"frames": 1, "out": str(config.out)},
    )
    markdown = format_gif_gallery_markdown(
        results,
        title="README GIFs",
        path_relative_to=tmp_path,
    )

    assert "README GIFs" in markdown
    assert "PyBullet simulation" in markdown
    assert "simulation_pick_red_block_dummy.gif" in markdown
    assert "vla-zoo demo pybullet --model dummy" in markdown


def test_write_gif_manifest_and_gallery(tmp_path: Path) -> None:
    specs = build_pybullet_gif_specs(
        models=("dummy",),
        tasks=resolve_pybullet_tasks("pick_red_block"),
        out_dir=tmp_path,
    )
    results = render_pybullet_gif_suite(
        specs,
        renderer=lambda config: {"frames": 3, "out": str(config.out)},
    )

    manifest = tmp_path / "manifest.json"
    gallery = tmp_path / "README.md"
    write_gif_manifest(manifest, results)
    write_gif_gallery(gallery, results)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema"] == "vla_zoo.pybullet_gif_suite.v1"
    assert payload["results"][0]["spec"]["model_name"] == "dummy"
    assert "PyBullet GIF Gallery" in gallery.read_text(encoding="utf-8")


def test_check_gif_suite_validates_manifest_and_links(tmp_path: Path) -> None:
    _write_test_manifest(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("![demo](demo.gif)\n", encoding="utf-8")

    report = check_gif_suite(
        tmp_path,
        expected_width=4,
        expected_height=3,
        min_frames=3,
        min_bytes=1,
        link_files=(readme,),
    )

    assert report.ok
    assert len(report.assets) == 1
    assert report.assets[0].model_name == "dummy"
    assert report.assets[0].task_id == "pick_red_block"
    assert report.assets[0].frames == 3
    assert report.assets[0].width == 4
    assert report.assets[0].height == 3


def test_check_gif_suite_detects_blank_gif(tmp_path: Path) -> None:
    gif_path = tmp_path / "blank.gif"
    Image.new("RGB", (4, 3), color=(0, 0, 0)).save(gif_path, save_all=True)
    result = PyBulletGifResult(
        spec=PyBulletGifSpec(
            model_name="dummy",
            task_id="pick_red_block",
            instruction="pick up the red block",
            out=gif_path,
        ),
        ok=True,
        frames=1,
        bytes=gif_path.stat().st_size,
    )
    write_gif_manifest(tmp_path / "gif_manifest.json", [result])

    report = check_gif_suite(
        tmp_path,
        expected_width=4,
        expected_height=3,
        min_frames=1,
        min_bytes=1,
    )

    assert not report.ok
    assert any(issue.code == "gif_low_variance" for issue in report.assets[0].issues)


def test_gif_check_markdown_and_html_report(tmp_path: Path) -> None:
    _write_test_manifest(tmp_path)
    report = check_gif_suite(
        tmp_path,
        expected_width=4,
        expected_height=3,
        min_frames=3,
        min_bytes=1,
    )
    markdown = format_gif_check_markdown(report)
    html = format_gif_report_html(report)

    assert "PyBullet GIF Check" in markdown
    assert "does not validate VLA model quality" in markdown
    assert "demo.gif" in markdown
    assert "vla_zoo GIF Gallery" in html
    assert "What This Proves" in html
    assert "What This Does Not Prove" in html
    assert "Task x Adapter Matrix" in html
    assert "pick red block" in html
    assert "dummy" in html
    assert '<img src="demo.gif"' in html


def test_write_gif_report_html(tmp_path: Path) -> None:
    _write_test_manifest(tmp_path)
    report = check_gif_suite(
        tmp_path,
        expected_width=4,
        expected_height=3,
        min_frames=3,
        min_bytes=1,
    )
    out = tmp_path / "index.html"

    write_gif_report_html(out, report, title="Test GIF Report")

    text = out.read_text(encoding="utf-8")
    assert "Test GIF Report" in text
    assert "demo.gif" in text


def test_cli_demo_gif_check_and_report(tmp_path: Path) -> None:
    manifest = _write_test_manifest(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("![demo](demo.gif)\n", encoding="utf-8")
    check_json = tmp_path / "check.json"
    html = tmp_path / "index.html"

    check_result = CliRunner().invoke(
        app,
        [
            "demo",
            "gif-check",
            str(tmp_path),
            "--expected-width",
            "4",
            "--expected-height",
            "3",
            "--min-frames",
            "3",
            "--min-bytes",
            "1",
            "--link-files",
            str(readme),
            "--out",
            str(check_json),
        ],
    )
    report_result = CliRunner().invoke(
        app,
        [
            "demo",
            "gif-report",
            "--manifest",
            str(manifest),
            "--expected-width",
            "4",
            "--expected-height",
            "3",
            "--min-frames",
            "3",
            "--min-bytes",
            "1",
            "--html-out",
            str(html),
            "--check-json-out",
            str(tmp_path / "report_check.json"),
            "--link-files",
            str(readme),
        ],
    )

    assert check_result.exit_code == 0
    assert report_result.exit_code == 0
    assert json.loads(check_json.read_text(encoding="utf-8"))["ok"]
    assert "demo.gif" in html.read_text(encoding="utf-8")
