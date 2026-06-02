from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from vla_zoo.demo.pybullet import (
    PyBulletDemoConfig,
    PyBulletTaskSpec,
    default_pybullet_tasks,
    pybullet_task_by_id,
    render_pybullet_demo,
)

GifRenderer = Callable[[PyBulletDemoConfig], dict[str, object]]


@dataclass(frozen=True)
class PyBulletGifSpec:
    """One README-oriented PyBullet GIF render target."""

    model_name: str
    task_id: str
    instruction: str
    out: Path
    runtime: str = "local"
    remote_url: str = "http://localhost:8000"
    model_call_every: int = 8
    render_stride: int = 8
    cube_initial_position: tuple[float, float, float] = (0.58, -0.16, 0.035)
    cube_goal_position: tuple[float, float, float] = (0.58, 0.22, 0.035)
    goal_tolerance_m: float = 0.15

    def to_config(self) -> PyBulletDemoConfig:
        return PyBulletDemoConfig(
            model_name=self.model_name,
            runtime=self.runtime,
            remote_url=self.remote_url,
            instruction=self.instruction,
            task_id=self.task_id,
            cube_initial_position=self.cube_initial_position,
            cube_goal_position=self.cube_goal_position,
            goal_tolerance_m=self.goal_tolerance_m,
            out=self.out,
            model_call_every=self.model_call_every,
            render_stride=self.render_stride,
        )

    def command(self) -> str:
        command = [
            "vla-zoo",
            "demo",
            "pybullet",
            "--model",
            self.model_name,
            "--task",
            self.task_id,
            "--out",
            str(self.out),
            "--model-call-every",
            str(self.model_call_every),
            "--render-stride",
            str(self.render_stride),
        ]
        if self.runtime != "local":
            command.extend(["--runtime", self.runtime, "--remote-url", self.remote_url])
        return " ".join(command)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["out"] = str(self.out)
        payload["command"] = self.command()
        return payload


@dataclass(frozen=True)
class PyBulletGifResult:
    spec: PyBulletGifSpec
    ok: bool
    frames: int = 0
    bytes: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec.to_dict(),
            "ok": self.ok,
            "frames": self.frames,
            "bytes": self.bytes,
            "error": self.error,
        }


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "item"


def resolve_pybullet_tasks(tasks: str | Sequence[str]) -> list[PyBulletTaskSpec]:
    raw_items = tasks.split(",") if isinstance(tasks, str) else list(tasks)
    items = [item.strip() for item in raw_items if item.strip()]
    if not items or "all" in {item.lower() for item in items}:
        return default_pybullet_tasks()
    return [pybullet_task_by_id(item) for item in items]


def build_pybullet_gif_specs(
    *,
    models: Sequence[str],
    tasks: Sequence[PyBulletTaskSpec],
    out_dir: Path,
    runtime: str = "local",
    remote_url: str = "http://localhost:8000",
    model_call_every: int = 8,
    render_stride: int = 8,
    filename_prefix: str = "simulation",
) -> list[PyBulletGifSpec]:
    specs: list[PyBulletGifSpec] = []
    for task in tasks:
        for model in models:
            filename = f"{filename_prefix}_{_slug(task.task_id)}_{_slug(model)}.gif"
            specs.append(
                PyBulletGifSpec(
                    model_name=model,
                    task_id=task.task_id,
                    instruction=task.instruction,
                    out=out_dir / filename,
                    runtime=runtime,
                    remote_url=remote_url,
                    model_call_every=model_call_every,
                    render_stride=render_stride,
                    cube_initial_position=task.cube_initial_position,
                    cube_goal_position=task.cube_goal_position,
                    goal_tolerance_m=task.goal_tolerance_m,
                )
            )
    return specs


def render_pybullet_gif_suite(
    specs: Sequence[PyBulletGifSpec],
    *,
    renderer: GifRenderer = render_pybullet_demo,
) -> list[PyBulletGifResult]:
    results: list[PyBulletGifResult] = []
    for spec in specs:
        try:
            rendered = renderer(spec.to_config())
            bytes_written = spec.out.stat().st_size if spec.out.exists() else 0
            results.append(
                PyBulletGifResult(
                    spec=spec,
                    ok=True,
                    frames=_object_to_int(rendered.get("frames"), default=0),
                    bytes=bytes_written,
                )
            )
        except Exception as exc:
            results.append(PyBulletGifResult(spec=spec, ok=False, error=str(exc)))
    return results


def _object_to_int(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return default


def write_gif_manifest(path: Path, results: Sequence[PyBulletGifResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "vla_zoo.pybullet_gif_suite.v1",
        "results": [result.to_dict() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _display_path(path: Path, *, relative_to: Path | None) -> str:
    if relative_to is None:
        return str(path)
    try:
        return path.relative_to(relative_to).as_posix()
    except ValueError:
        return str(path)


def format_gif_gallery_markdown(
    results: Sequence[PyBulletGifResult],
    *,
    title: str = "PyBullet GIF Gallery",
    columns: int = 3,
    path_relative_to: Path | None = None,
) -> str:
    columns = max(1, columns)
    lines = [
        f"## {title}",
        "",
        "These GIFs are rendered from the bundled PyBullet simulation. They show runtime "
        "plumbing and adapter action traces, not real robot task success.",
        "",
    ]
    for start in range(0, len(results), columns):
        row = list(results[start : start + columns])
        headers = [f"{item.spec.model_name} / {item.spec.task_id}" for item in row]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---" for _ in row) + "|")
        cells: list[str] = []
        for result in row:
            path = _display_path(result.spec.out, relative_to=path_relative_to)
            alt = f"{result.spec.model_name} {result.spec.task_id} PyBullet simulation"
            if result.ok:
                caption = f"frames={result.frames}, size={result.bytes} bytes"
                cells.append(f"![{alt}]({path})<br>{caption}")
            else:
                cells.append(f"render failed: `{result.error or 'unknown error'}`")
        lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(
        [
            "Reproduce a single GIF:",
            "",
            "```bash",
            results[0].spec.command() if results else "vla-zoo demo pybullet --model dummy",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def write_gif_gallery(path: Path, results: Sequence[PyBulletGifResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_gif_gallery_markdown(results, path_relative_to=path.parent),
        encoding="utf-8",
    )
