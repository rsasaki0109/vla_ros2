from __future__ import annotations

import json
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from vla_zoo.benchmark.runner import run_smoke_benchmark
from vla_zoo.core.errors import MissingDependencyError, VLAZooError
from vla_zoo.core.registry import get_adapter_info, list_models, load_model
from vla_zoo.core.types import VLAAction, VLAActionChunk

if TYPE_CHECKING:
    from vla_zoo.demo.pybullet import PyBulletComparisonTarget

app = typer.Typer(
    help="ROS2-native runtime, benchmark, and adapter hub for Vision-Language-Action models.",
    no_args_is_help=True,
)
demo_app = typer.Typer(help="Generate runnable demos.")
compare_app = typer.Typer(help="Compare VLA adapters and runtime paths.")
app.add_typer(demo_app, name="demo")
app.add_typer(compare_app, name="compare")


def _adapter_status(name: str) -> str:
    if name == "dummy":
        return "available"
    if name == "openvla":
        missing = [dep for dep in ("torch", "transformers") if find_spec(dep) is None]
        if missing:
            return 'missing optional deps: pip install "vla_zoo[openvla]"'
        return "available"
    info = get_adapter_info(name)
    return "experimental" if info.experimental else "available"


def _format_optional_float(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:0.2f}{suffix}"


def _shorten(value: str | None, limit: int = 56) -> str:
    if not value:
        return "-"
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _parse_remote_map(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    parsed: dict[str, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        model, separator, url = item.partition("=")
        if not separator or not model.strip() or not url.strip():
            raise typer.BadParameter(
                "Remote map entries must use model=url, for example "
                "openvla=http://gpu-box:8001"
            )
        parsed[model.strip().lower()] = url.strip()
    return parsed


def _parse_paths(value: str) -> list[Path]:
    paths = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not paths:
        raise typer.BadParameter("At least one result JSON path is required.")
    return paths


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise typer.BadParameter(f"Could not read manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Manifest {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("Comparison manifest must be a JSON object.")
    return payload


def _manifest_string(payload: dict[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str):
        raise typer.BadParameter(f"Manifest field {key!r} must be a string.")
    return value


def _manifest_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int):
        raise typer.BadParameter(f"Manifest field {key!r} must be an integer.")
    return value


def _manifest_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise typer.BadParameter(f"Manifest field {key!r} must be a boolean.")
    return value


def _manifest_output_path(
    payload: dict[str, Any],
    output_name: str,
    explicit_path: Path | None,
) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    outputs = payload.get("outputs", {})
    if outputs == {}:
        return None
    if not isinstance(outputs, dict):
        raise typer.BadParameter("Manifest field 'outputs' must be an object.")
    value = outputs.get(output_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise typer.BadParameter(f"Manifest output {output_name!r} must be a string path.")
    return Path(value)


def _manifest_targets(payload: dict[str, Any]) -> list[PyBulletComparisonTarget]:
    from vla_zoo.demo.pybullet import PyBulletComparisonTarget

    raw_targets = payload.get("models", payload.get("targets"))
    if not isinstance(raw_targets, list) or not raw_targets:
        raise typer.BadParameter("Comparison manifest must contain a non-empty 'models' list.")

    default_runtime = _manifest_string(payload, "runtime", "local")
    default_remote_url = _manifest_string(payload, "remote_url", "http://localhost:8000")
    targets: list[PyBulletComparisonTarget] = []
    for index, raw_target in enumerate(raw_targets):
        if isinstance(raw_target, str):
            targets.append(
                PyBulletComparisonTarget(
                    model_name=raw_target,
                    runtime=default_runtime,
                    remote_url=default_remote_url,
                )
            )
            continue
        if not isinstance(raw_target, dict):
            raise typer.BadParameter(f"Manifest model entry {index} must be a string or object.")

        name = raw_target.get("name", raw_target.get("model"))
        if not isinstance(name, str) or not name.strip():
            raise typer.BadParameter(f"Manifest model entry {index} needs a non-empty name.")
        runtime = raw_target.get("runtime", default_runtime)
        remote_url = raw_target.get("remote_url", default_remote_url)
        adapter_kwargs = raw_target.get("adapter_kwargs")
        if not isinstance(runtime, str):
            raise typer.BadParameter(f"Manifest model entry {index} runtime must be a string.")
        if not isinstance(remote_url, str):
            raise typer.BadParameter(f"Manifest model entry {index} remote_url must be a string.")
        if adapter_kwargs is not None and not isinstance(adapter_kwargs, dict):
            raise typer.BadParameter(
                f"Manifest model entry {index} adapter_kwargs must be an object."
            )
        targets.append(
            PyBulletComparisonTarget(
                model_name=name,
                runtime=runtime,
                remote_url=remote_url,
                adapter_kwargs=dict(adapter_kwargs) if adapter_kwargs is not None else None,
            )
        )
    return targets


@app.command("list")
def list_command() -> None:
    """List registered model adapters."""

    for info in list_models():
        typer.echo(f"{info.name:<11} {info.source:<10} {_adapter_status(info.name)}")


@app.command()
def info(model: str) -> None:
    """Show adapter metadata."""

    adapter = get_adapter_info(model)
    typer.echo(
        json.dumps(
            {
                "name": adapter.name,
                "source": adapter.source,
                "aliases": adapter.aliases,
                "experimental": adapter.experimental,
                "domain": adapter.domain,
                "description": adapter.description,
                "install_hint": adapter.install_hint,
            },
            indent=2,
        )
    )


@app.command()
def predict(
    model: Annotated[str, typer.Option("--model", "-m")] = "dummy",
    instruction: Annotated[str, typer.Option("--instruction", "-i")] = "test",
) -> None:
    """Run one prediction and print JSON."""

    try:
        loaded = load_model(model)
        action = loaded.predict(image=None, instruction=instruction)
    except VLAZooError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    payload: dict[str, object]
    if isinstance(action, VLAActionChunk):
        payload = {
            "kind": "chunk",
            "actions": [
                {
                    "action_space": item.spec.action_space,
                    "data": item.tolist(),
                    "dt": item.dt,
                    "confidence": item.confidence,
                    "metadata": item.metadata,
                }
                for item in action.actions
            ],
        }
    elif isinstance(action, VLAAction):
        payload = {
            "action_space": action.spec.action_space,
            "data": action.tolist(),
            "dt": action.dt,
            "confidence": action.confidence,
            "metadata": action.metadata,
        }
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def serve(
    model: Annotated[str, typer.Option("--model", "-m")] = "dummy",
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port")] = 8000,
) -> None:
    """Start the optional FastAPI inference server."""

    from vla_zoo.runtime.server import run_server

    try:
        run_server(model_name=model, host=host, port=port)
    except MissingDependencyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command()
def bench(
    model: Annotated[str, typer.Option("--model", "-m")] = "dummy",
    benchmark: Annotated[str, typer.Option("--benchmark", "-b")] = "smoke",
    episodes: Annotated[int, typer.Option("--episodes", "-e")] = 3,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Run a benchmark."""

    if benchmark != "smoke":
        raise typer.BadParameter("Only the smoke benchmark is implemented in the MVP.")
    loaded = load_model(model)
    typer.echo(json.dumps(run_smoke_benchmark(loaded, episodes=episodes, seed=seed), indent=2))


@compare_app.command("adapters")
def compare_adapters() -> None:
    """Compare registered adapter metadata and availability."""

    typer.echo(f"{'model':<11} {'status':<54} {'domain':<22} aliases")
    typer.echo(f"{'-' * 11} {'-' * 54} {'-' * 22} {'-' * 24}")
    for adapter in list_models():
        aliases = ", ".join(adapter.aliases) if adapter.aliases else "-"
        typer.echo(
            f"{adapter.name:<11} "
            f"{_shorten(_adapter_status(adapter.name), 54):<54} "
            f"{(adapter.domain or '-'):<22} "
            f"{aliases}"
        )


@compare_app.command("dashboard")
def compare_dashboard(
    results: Annotated[
        str,
        typer.Option(
            "--results",
            "-r",
            help="Comma-separated comparison JSON result paths.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output HTML dashboard path."),
    ] = Path("results/vla_runtime_dashboard.html"),
    title: Annotated[
        str,
        typer.Option("--title", help="Dashboard title."),
    ] = "vla_zoo Runtime Dashboard",
) -> None:
    """Build an interactive static dashboard from comparison result JSON."""

    from vla_zoo.runtime.dashboard import (
        format_comparison_dashboard_html,
        load_dashboard_records,
    )

    try:
        records = load_dashboard_records(_parse_paths(results))
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    _write_text(out, format_comparison_dashboard_html(records, title=title))
    typer.echo(f"Dashboard written to {out}")


@compare_app.command("pybullet")
def compare_pybullet(
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated adapters to compare in the same PyBullet smoke scene.",
        ),
    ] = "dummy,openvla,pi0,smolvla,groot",
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            help="JSON manifest with per-model runtime and endpoint settings.",
        ),
    ] = None,
    runtime: Annotated[str, typer.Option("--runtime")] = "local",
    remote_url: Annotated[str, typer.Option("--remote-url")] = "http://localhost:8000",
    remote_map: Annotated[
        str | None,
        typer.Option(
            "--remote-map",
            help=(
                "Comma-separated model=url overrides for remote comparisons, "
                "for example openvla=http://gpu:8001,pi0=http://gpu:8002."
            ),
        ),
    ] = None,
    instruction: Annotated[
        str,
        typer.Option("--instruction", "-i"),
    ] = "pick up the red block",
    model_call_every: Annotated[int, typer.Option("--model-call-every")] = 8,
    render_stride: Annotated[int, typer.Option("--render-stride")] = 12,
    allow_local_heavy: Annotated[
        bool,
        typer.Option(
            "--allow-local-heavy",
            help="Allow local heavy adapters such as OpenVLA to load real model weights.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON instead of a table."),
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write JSON results to this path."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write a README-ready Markdown table."),
    ] = None,
    html_out: Annotated[
        Path | None,
        typer.Option("--html-out", help="Write a self-contained HTML comparison report."),
    ] = None,
) -> None:
    """Run one deterministic PyBullet smoke scene per adapter and compare runtime metrics."""

    from vla_zoo.demo.pybullet import (
        compare_pybullet_models,
        compare_pybullet_targets,
        format_pybullet_comparison_html,
        format_pybullet_comparison_markdown,
    )

    markdown_title = "PyBullet VLA Runtime Comparison"
    if manifest is not None:
        manifest_payload = _load_json_manifest(manifest)
        targets = _manifest_targets(manifest_payload)
        instruction = _manifest_string(manifest_payload, "instruction", instruction)
        model_call_every = _manifest_int(
            manifest_payload,
            "model_call_every",
            model_call_every,
        )
        render_stride = _manifest_int(manifest_payload, "render_stride", render_stride)
        allow_local_heavy = _manifest_bool(
            manifest_payload,
            "allow_local_heavy",
            allow_local_heavy,
        )
        markdown_title = _manifest_string(manifest_payload, "title", markdown_title)
        out = _manifest_output_path(manifest_payload, "json", out)
        markdown_out = _manifest_output_path(manifest_payload, "markdown", markdown_out)
        html_out = _manifest_output_path(manifest_payload, "html", html_out)
        results = compare_pybullet_targets(
            targets,
            instruction=instruction,
            model_call_every=model_call_every,
            render_stride=render_stride,
            allow_local_heavy=allow_local_heavy,
        )
    else:
        model_names = [item.strip() for item in models.split(",") if item.strip()]
        if not model_names:
            raise typer.BadParameter("At least one model name is required.")
        remote_urls = _parse_remote_map(remote_map)
        results = compare_pybullet_models(
            model_names,
            runtime=runtime,
            remote_url=remote_url,
            remote_urls=remote_urls or None,
            instruction=instruction,
            model_call_every=model_call_every,
            render_stride=render_stride,
            allow_local_heavy=allow_local_heavy,
        )

    json_payload = json.dumps([asdict(result) for result in results], indent=2)
    if out is not None:
        _write_text(out, f"{json_payload}\n")
    if markdown_out is not None:
        _write_text(
            markdown_out,
            format_pybullet_comparison_markdown(results, title=markdown_title),
        )
    if html_out is not None:
        _write_text(
            html_out,
            format_pybullet_comparison_html(results, title=markdown_title),
        )
    if json_output:
        typer.echo(json_payload)
        return

    typer.echo(
        f"{'model':<11} {'ok':<5} {'frames':>6} {'queries':>7} {'errors':>6} "
        f"{'mean_ms':>9} {'max_ms':>9} {'mean|a|':>9} note"
    )
    typer.echo(
        f"{'-' * 11} {'-' * 5} {'-' * 6} {'-' * 7} {'-' * 6} "
        f"{'-' * 9} {'-' * 9} {'-' * 9} {'-' * 32}"
    )
    for result in results:
        typer.echo(
            f"{result.model_name:<11} "
            f"{str(result.ok):<5} "
            f"{result.frames:>6} "
            f"{result.adapter_queries:>7} "
            f"{result.adapter_errors:>6} "
            f"{_format_optional_float(result.mean_latency_ms):>9} "
            f"{_format_optional_float(result.max_latency_ms):>9} "
            f"{_format_optional_float(result.mean_abs_action):>9} "
            f"{_shorten(result.last_error)}"
        )
    if out is not None:
        typer.echo(f"\nJSON written to {out}")
    if markdown_out is not None:
        typer.echo(f"Markdown written to {markdown_out}")
    if html_out is not None:
        typer.echo(f"HTML written to {html_out}")


@demo_app.command("pybullet")
def demo_pybullet(
    model: Annotated[str, typer.Option("--model", "-m")] = "dummy",
    runtime: Annotated[str, typer.Option("--runtime")] = "local",
    remote_url: Annotated[str, typer.Option("--remote-url")] = "http://localhost:8000",
    instruction: Annotated[str, typer.Option("--instruction", "-i")] = "pick up the red block",
    out: Annotated[Path, typer.Option("--out", "-o")] = Path(
        "docs/assets/simulation_pick_place.gif"
    ),
    model_call_every: Annotated[int, typer.Option("--model-call-every")] = 8,
) -> None:
    """Render the PyBullet pick-and-place demo with any VLA adapter."""

    from vla_zoo.demo.pybullet import PyBulletDemoConfig, render_pybullet_demo

    try:
        result = render_pybullet_demo(
            PyBulletDemoConfig(
                model_name=model,
                runtime=runtime,
                remote_url=remote_url,
                instruction=instruction,
                out=out,
                model_call_every=model_call_every,
            )
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
