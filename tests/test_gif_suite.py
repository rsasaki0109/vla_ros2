from __future__ import annotations

import json
from pathlib import Path

from vla_zoo.demo.gif_suite import (
    build_pybullet_gif_specs,
    format_gif_gallery_markdown,
    render_pybullet_gif_suite,
    resolve_pybullet_tasks,
    write_gif_gallery,
    write_gif_manifest,
)
from vla_zoo.demo.pybullet import PyBulletDemoConfig


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
