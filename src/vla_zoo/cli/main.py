from __future__ import annotations

import json
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated

import typer

from vla_zoo.benchmark.runner import run_smoke_benchmark
from vla_zoo.core.errors import MissingDependencyError, VLAZooError
from vla_zoo.core.registry import get_adapter_info, list_models, load_model
from vla_zoo.core.types import VLAAction, VLAActionChunk

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
    runtime: Annotated[str, typer.Option("--runtime")] = "local",
    remote_url: Annotated[str, typer.Option("--remote-url")] = "http://localhost:8000",
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
) -> None:
    """Run one deterministic PyBullet smoke scene per adapter and compare runtime metrics."""

    from vla_zoo.demo.pybullet import compare_pybullet_models

    model_names = [item.strip() for item in models.split(",") if item.strip()]
    if not model_names:
        raise typer.BadParameter("At least one model name is required.")

    results = compare_pybullet_models(
        model_names,
        runtime=runtime,
        remote_url=remote_url,
        instruction=instruction,
        model_call_every=model_call_every,
        render_stride=render_stride,
        allow_local_heavy=allow_local_heavy,
    )
    if json_output:
        typer.echo(json.dumps([asdict(result) for result in results], indent=2))
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
