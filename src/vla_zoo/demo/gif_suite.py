from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

from PIL import Image, ImageSequence, ImageStat

from vla_zoo.demo.pybullet import (
    PyBulletDemoConfig,
    PyBulletTaskSpec,
    default_pybullet_tasks,
    pybullet_task_by_id,
    render_pybullet_demo,
)

GifRenderer = Callable[[PyBulletDemoConfig], dict[str, object]]
LOCAL_LINK_RE = re.compile(
    r"""(?:!\[[^\]]*\]\(([^)]+)\)|\[[^\]]+\]\(([^)]+)\)|(?:href|src)=["']([^"']+)["'])"""
)


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


@dataclass(frozen=True)
class GifCheckIssue:
    level: str
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GifAssetCheck:
    path: str
    ok: bool
    frames: int
    width: int
    height: int
    bytes: int
    model_name: str | None = None
    task_id: str | None = None
    instruction: str | None = None
    command: str | None = None
    issues: tuple[GifCheckIssue, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GifSuiteCheckReport:
    manifest: str
    ok: bool
    assets: tuple[GifAssetCheck, ...]
    issues: tuple[GifCheckIssue, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def _manifest_spec(result: dict[str, object]) -> dict[str, object]:
    raw_spec = result.get("spec")
    if not isinstance(raw_spec, dict):
        return {}
    return raw_spec


def _manifest_spec_string(result: dict[str, object], key: str) -> str | None:
    raw_value = _manifest_spec(result).get(key)
    return raw_value if isinstance(raw_value, str) else None


def _manifest_spec_out(result: dict[str, object]) -> str | None:
    raw_out = _manifest_spec(result).get("out")
    return raw_out if isinstance(raw_out, str) else None


def _resolve_asset_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    candidate = manifest_path.parent / path.name
    if candidate.exists():
        return candidate
    return path


def _local_links_from_text(text: str) -> list[str]:
    links: list[str] = []
    for match in LOCAL_LINK_RE.finditer(text):
        raw = next((group for group in match.groups() if group), "")
        link = raw.split("#", 1)[0].strip()
        if not link or link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if link.endswith(".gif") or "gif_suite" in link:
            links.append(link)
    return links


def _check_link_file(path: Path) -> list[GifCheckIssue]:
    issues: list[GifCheckIssue] = []
    if not path.exists():
        return [
            GifCheckIssue(
                level="warning",
                code="link_file_missing",
                message=f"link file does not exist: {path}",
                path=str(path),
            )
        ]
    text = path.read_text(encoding="utf-8")
    for link in _local_links_from_text(text):
        target = path.parent / link
        if not target.exists():
            issues.append(
                GifCheckIssue(
                    level="error",
                    code="broken_link",
                    message=f"local link target does not exist: {link}",
                    path=str(path),
                )
            )
    return issues


def _is_low_variance_frame(frame: Image.Image, *, threshold: int) -> bool:
    stat = ImageStat.Stat(frame.convert("RGB").resize((32, 18)))
    ranges = [int(high - low) for low, high in stat.extrema]
    return max(ranges, default=0) <= threshold


def _inspect_gif(path: Path, *, blank_threshold: int) -> tuple[int, int, int, bool]:
    with Image.open(path) as image:
        width, height = image.size
        frame_count = getattr(image, "n_frames", 1)
        sample_indices = {0, max(0, frame_count // 2), max(0, frame_count - 1)}
        low_variance = True
        for index, frame in enumerate(ImageSequence.Iterator(image)):
            if index not in sample_indices:
                continue
            if not _is_low_variance_frame(frame, threshold=blank_threshold):
                low_variance = False
                break
        return int(frame_count), int(width), int(height), low_variance


def check_gif_suite(
    path: Path,
    *,
    expected_width: int = 960,
    expected_height: int = 540,
    min_frames: int = 12,
    min_bytes: int = 1024,
    blank_threshold: int = 4,
    link_files: Sequence[Path] = (),
) -> GifSuiteCheckReport:
    """Validate generated GIFs, manifest consistency, and optional README/Page links."""

    manifest_path = path if path.is_file() else path / "gif_manifest.json"
    issues: list[GifCheckIssue] = []
    assets: list[GifAssetCheck] = []
    if not manifest_path.exists():
        issue = GifCheckIssue(
            level="error",
            code="manifest_missing",
            message=f"manifest does not exist: {manifest_path}",
            path=str(manifest_path),
        )
        return GifSuiteCheckReport(str(manifest_path), ok=False, assets=(), issues=(issue,))

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issue = GifCheckIssue(
            level="error",
            code="manifest_invalid_json",
            message=str(exc),
            path=str(manifest_path),
        )
        return GifSuiteCheckReport(str(manifest_path), ok=False, assets=(), issues=(issue,))

    raw_results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        issues.append(
            GifCheckIssue(
                level="error",
                code="manifest_results_missing",
                message="manifest does not contain a results list",
                path=str(manifest_path),
            )
        )
        raw_results = []

    manifest_paths: set[Path] = set()
    for index, raw_result in enumerate(raw_results):
        result = raw_result if isinstance(raw_result, dict) else {}
        asset_metadata = {
            "model_name": _manifest_spec_string(result, "model_name"),
            "task_id": _manifest_spec_string(result, "task_id"),
            "instruction": _manifest_spec_string(result, "instruction"),
            "command": _manifest_spec_string(result, "command"),
        }
        raw_out = _manifest_spec_out(result)
        if raw_out is None:
            issue = GifCheckIssue(
                level="error",
                code="manifest_out_missing",
                message=f"manifest result {index} does not declare spec.out",
                path=str(manifest_path),
            )
            assets.append(
                GifAssetCheck(
                    path=f"manifest[{index}]",
                    ok=False,
                    frames=0,
                    width=0,
                    height=0,
                    bytes=0,
                    **asset_metadata,
                    issues=(issue,),
                )
            )
            continue

        asset_path = _resolve_asset_path(manifest_path, raw_out)
        manifest_paths.add(asset_path)
        asset_issues: list[GifCheckIssue] = []
        if not asset_path.exists():
            asset_issues.append(
                GifCheckIssue(
                    level="error",
                    code="gif_missing",
                    message="GIF file does not exist",
                    path=str(asset_path),
                )
            )
            assets.append(
                GifAssetCheck(
                    str(asset_path),
                    ok=False,
                    frames=0,
                    width=0,
                    height=0,
                    bytes=0,
                    **asset_metadata,
                    issues=tuple(asset_issues),
                )
            )
            continue

        size = asset_path.stat().st_size
        if size < min_bytes:
            asset_issues.append(
                GifCheckIssue(
                    level="error",
                    code="gif_too_small",
                    message=f"GIF size {size} is below minimum {min_bytes}",
                    path=str(asset_path),
                )
            )

        try:
            frames, width, height, low_variance = _inspect_gif(
                asset_path,
                blank_threshold=blank_threshold,
            )
        except OSError as exc:
            asset_issues.append(
                GifCheckIssue(
                    level="error",
                    code="gif_decode_error",
                    message=str(exc),
                    path=str(asset_path),
                )
            )
            assets.append(
                GifAssetCheck(
                    str(asset_path),
                    ok=False,
                    frames=0,
                    width=0,
                    height=0,
                    bytes=size,
                    **asset_metadata,
                    issues=tuple(asset_issues),
                )
            )
            continue

        if frames < min_frames:
            asset_issues.append(
                GifCheckIssue(
                    level="error",
                    code="gif_too_few_frames",
                    message=f"GIF has {frames} frames; expected at least {min_frames}",
                    path=str(asset_path),
                )
            )
        if width != expected_width or height != expected_height:
            asset_issues.append(
                GifCheckIssue(
                    level="error",
                    code="gif_wrong_resolution",
                    message=(
                        f"GIF resolution is {width}x{height}; "
                        f"expected {expected_width}x{expected_height}"
                    ),
                    path=str(asset_path),
                )
            )
        if low_variance:
            asset_issues.append(
                GifCheckIssue(
                    level="error",
                    code="gif_low_variance",
                    message="sampled GIF frames have very low pixel variance",
                    path=str(asset_path),
                )
            )

        manifest_bytes = result.get("bytes")
        if isinstance(manifest_bytes, int) and manifest_bytes != size:
            asset_issues.append(
                GifCheckIssue(
                    level="warning",
                    code="manifest_size_mismatch",
                    message=f"manifest bytes={manifest_bytes}; actual bytes={size}",
                    path=str(asset_path),
                )
            )
        assets.append(
            GifAssetCheck(
                str(asset_path),
                ok=not any(issue.level == "error" for issue in asset_issues),
                frames=frames,
                width=width,
                height=height,
                bytes=size,
                **asset_metadata,
                issues=tuple(asset_issues),
            )
        )

    search_dir = manifest_path.parent
    for extra in sorted(search_dir.glob("*.gif")):
        if extra not in manifest_paths:
            issues.append(
                GifCheckIssue(
                    level="warning",
                    code="gif_not_in_manifest",
                    message=f"GIF file is not listed in manifest: {extra.name}",
                    path=str(extra),
                )
            )
    for link_file in link_files:
        issues.extend(_check_link_file(link_file))

    has_error = any(issue.level == "error" for issue in issues) or any(
        issue.level == "error" for asset in assets for issue in asset.issues
    )
    return GifSuiteCheckReport(
        manifest=str(manifest_path),
        ok=not has_error,
        assets=tuple(assets),
        issues=tuple(issues),
    )


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


def format_gif_check_markdown(report: GifSuiteCheckReport) -> str:
    lines = [
        "## PyBullet GIF Check",
        "",
        f"- manifest: `{report.manifest}`",
        f"- status: {'ok' if report.ok else 'failed'}",
        f"- assets: {len(report.assets)}",
        "",
        "This QA report checks that generated GIFs exist, decode, have enough frames, "
        "use the expected resolution, are not low-variance blank files, and match the "
        "manifest. It does not validate VLA model quality or real robot behavior.",
        "",
        "| GIF | Status | Frames | Resolution | Size | Issues |",
        "|---|---|---:|---|---:|---|",
    ]
    for asset in report.assets:
        issues = "<br>".join(issue.message for issue in asset.issues) or "-"
        lines.append(
            f"| `{Path(asset.path).name}` | {'ok' if asset.ok else 'failed'} | "
            f"{asset.frames} | {asset.width}x{asset.height} | {asset.bytes} | {issues} |"
        )
    if report.issues:
        lines.extend(["", "### Suite Issues", ""])
        for issue in report.issues:
            lines.append(f"- {issue.level}: {issue.message}")
    return "\n".join(lines) + "\n"


def _human_label(value: str | None, *, fallback: str) -> str:
    if not value:
        return fallback
    return value.replace("_", " ")


def _asset_task_id(asset: GifAssetCheck) -> str:
    if asset.task_id:
        return asset.task_id
    stem = Path(asset.path).stem
    if stem.startswith("simulation_"):
        parts = stem.split("_")
        if len(parts) > 2:
            return "_".join(parts[1:-1])
    return "unknown_task"


def _asset_model_name(asset: GifAssetCheck) -> str:
    if asset.model_name:
        return asset.model_name
    stem = Path(asset.path).stem
    if stem.startswith("simulation_"):
        parts = stem.split("_")
        if len(parts) > 2:
            return parts[-1]
    return "unknown_model"


def _ordered_unique(values: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _asset_src(asset: GifAssetCheck, *, relative_to: Path) -> str:
    path = Path(asset.path)
    try:
        return path.relative_to(relative_to).as_posix()
    except ValueError:
        return str(path)


def _format_size(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} MB"
    if value >= 1_000:
        return f"{value / 1_000:.1f} KB"
    return f"{value} B"


def _format_gif_comparison_matrix(report: GifSuiteCheckReport, *, manifest_path: Path) -> str:
    if not report.assets:
        return ""
    tasks = _ordered_unique([_asset_task_id(asset) for asset in report.assets])
    models = _ordered_unique([_asset_model_name(asset) for asset in report.assets])
    by_cell = {(_asset_task_id(asset), _asset_model_name(asset)): asset for asset in report.assets}
    header_cells = "".join(
        f"<th>{escape(_human_label(model, fallback='model'))}</th>" for model in models
    )
    rows: list[str] = []
    for task in tasks:
        cells: list[str] = []
        for model in models:
            asset = by_cell.get((task, model))
            if asset is None:
                cells.append('<td><span class="missing">missing</span></td>')
                continue
            src = _asset_src(asset, relative_to=manifest_path.parent)
            status = "ok" if asset.ok else "failed"
            instruction = f"<p>{escape(asset.instruction)}</p>" if asset.instruction else ""
            cells.append(
                "<td>"
                f'<img src="{escape(src)}" '
                f'alt="{escape(model)} {escape(task)} PyBullet simulation">'
                f'<span class="status {status}">{status}</span>'
                f"{instruction}"
                f"<small>{asset.frames} frames &middot; {asset.width}x{asset.height} &middot; "
                f"{escape(_format_size(asset.bytes))}</small>"
                "</td>"
            )
        rows.append(
            "<tr>"
            f"<th>{escape(_human_label(task, fallback='task'))}</th>"
            f"{''.join(cells)}"
            "</tr>"
        )
    return (
        '<section class="matrix-section">'
        "<h2>Task x Adapter Matrix</h2>"
        "<p>Each cell is a checked GIF generated from the same PyBullet runtime boundary.</p>"
        '<div class="matrix-scroll">'
        '<table class="matrix">'
        f"<thead><tr><th>Task</th>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
        "</section>"
    )


def format_gif_report_html(
    report: GifSuiteCheckReport,
    *,
    title: str = "vla_zoo GIF Gallery",
) -> str:
    manifest_path = Path(report.manifest)
    task_count = len(_ordered_unique([_asset_task_id(asset) for asset in report.assets]))
    model_count = len(_ordered_unique([_asset_model_name(asset) for asset in report.assets]))
    matrix_html = _format_gif_comparison_matrix(report, manifest_path=manifest_path)
    cards: list[str] = []
    for asset in report.assets:
        path = Path(asset.path)
        src = _asset_src(asset, relative_to=manifest_path.parent)
        issues = "".join(
            f"<li>{escape(issue.level)}: {escape(issue.message)}</li>" for issue in asset.issues
        )
        issue_html = f"<ul>{issues}</ul>" if issues else "<p>No issues.</p>"
        label = (
            f"{_human_label(asset.model_name, fallback=path.stem)} / "
            f"{_human_label(asset.task_id, fallback='task')}"
        )
        command_html = (
            f"<dt>Command</dt><dd><code>{escape(asset.command)}</code></dd>"
            if asset.command
            else ""
        )
        cards.append(
            f"""
            <article class="card {'ok' if asset.ok else 'bad'}">
              <img src="{escape(src)}" alt="{escape(path.stem)} PyBullet simulation">
              <h2>{escape(label)}</h2>
              <dl>
                <dt>Status</dt><dd>{'ok' if asset.ok else 'failed'}</dd>
                <dt>Frames</dt><dd>{asset.frames}</dd>
                <dt>Resolution</dt><dd>{asset.width}x{asset.height}</dd>
                <dt>Size</dt><dd>{escape(_format_size(asset.bytes))}</dd>
                {command_html}
              </dl>
              {issue_html}
            </article>
            """
        )
    suite_issues = "".join(
        f"<li>{escape(issue.level)}: {escape(issue.message)}</li>" for issue in report.issues
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; }}
    body {{ margin: 0; background: #f8fafc; color: #111827; }}
    header {{ padding: 32px min(6vw, 72px) 18px; background: #111827; color: #f9fafb; }}
    main {{ padding: 28px min(6vw, 72px) 48px; }}
    .summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px; }}
    .pill {{ border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px 10px; background: #fff; }}
    .truth {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px; margin: 22px 0; }}
    .truth-card {{ background: #fff; border: 1px solid #dbe3ef; border-radius: 8px;
      padding: 16px; }}
    .matrix-section {{ margin: 24px 0; }}
    .matrix-section > p {{ color: #64748b; margin-top: -2px; }}
    .matrix-scroll {{ overflow-x: auto; border: 1px solid #dbe3ef; border-radius: 8px;
      background: #fff; }}
    .matrix {{ width: 100%; min-width: 820px; border-collapse: collapse; }}
    .matrix th, .matrix td {{ border-bottom: 1px solid #dbe3ef; border-right: 1px solid #dbe3ef;
      padding: 12px; vertical-align: top; text-align: left; }}
    .matrix th {{ background: #eef4f8; color: #111827; }}
    .matrix td {{ width: 28%; }}
    .matrix img {{ margin-bottom: 8px; }}
    .matrix small {{ display: block; color: #64748b; margin-top: 6px; }}
    .matrix p {{ margin: 4px 0; font-size: 13px; color: #334155; }}
    .status {{ display: inline-block; border-radius: 999px; padding: 3px 8px;
      color: #fff; font-size: 12px; font-weight: 800; text-transform: uppercase; }}
    .status.ok {{ background: #16a34a; }}
    .status.failed {{ background: #dc2626; }}
    .missing {{ color: #64748b; font-style: italic; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px; }}
    .card {{ background: #fff; border: 1px solid #dbe3ef; border-radius: 8px; padding: 14px; }}
    .card.ok {{ border-top: 4px solid #16a34a; }}
    .card.bad {{ border-top: 4px solid #dc2626; }}
    img {{ width: 100%; aspect-ratio: 16 / 9; object-fit: cover; border-radius: 6px; }}
    h1 {{ margin: 0; font-size: clamp(28px, 4vw, 42px); }}
    h2 {{ font-size: 16px; margin: 12px 0; }}
    dl {{ display: grid; grid-template-columns: 92px 1fr; gap: 4px 10px; margin: 0; }}
    dt {{ color: #64748b; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    code {{ font-size: 12px; }}
    ul {{ padding-left: 20px; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <p>Real PyBullet simulation GIFs generated by <code>vla-zoo demo gif-suite</code>.</p>
  </header>
  <main>
    <section class="summary">
      <div class="pill">status: <strong>{'ok' if report.ok else 'failed'}</strong></div>
      <div class="pill">assets: <strong>{len(report.assets)}</strong></div>
      <div class="pill">tasks: <strong>{task_count}</strong></div>
      <div class="pill">adapters: <strong>{model_count}</strong></div>
      <div class="pill">manifest: <code>{escape(Path(report.manifest).name)}</code></div>
    </section>
    <section class="truth">
      <div class="truth-card">
        <h2>What This Proves</h2>
        <ul>
          <li>PyBullet is actually simulated and rendered.</li>
          <li>Adapters receive rendered RGB observations and simulation state.</li>
          <li>The runtime calls adapter.predict() and records VLAAction traces.</li>
          <li>GIFs, manifests, and QA reports are reproducible.</li>
        </ul>
      </div>
      <div class="truth-card">
        <h2>What This Does Not Prove</h2>
        <ul>
          <li>Real robot task success.</li>
          <li>Zero-shot VLA policy quality.</li>
          <li>OpenVLA/pi0/SmolVLA benchmark performance.</li>
          <li>Hardware safety or calibrated robot deployment.</li>
        </ul>
      </div>
    </section>
    {f'<section><h2>Suite Issues</h2><ul>{suite_issues}</ul></section>' if suite_issues else ''}
    {matrix_html}
    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""
    return "\n".join(line.rstrip() for line in html.splitlines()) + "\n"


def write_gif_report_html(
    path: Path,
    report: GifSuiteCheckReport,
    *,
    title: str = "vla_zoo GIF Gallery",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_gif_report_html(report, title=title), encoding="utf-8")
