"""Minimal off-robot CLI for vla_ros2.

The CLI is intentionally small: it exists to sanity-check adapter loading and
inference outside of ROS2. The real runtime entry point is the ROS2 node
(``ros2 launch vla_ros2 <model>.launch.py``).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from vla_ros2 import list_models, load_model
from vla_ros2.compare import compare_models
from vla_ros2.core.errors import VLARos2Error

app = typer.Typer(
    add_completion=False,
    help="vla_ros2 — ROS2 real-robot VLA runtime (off-robot sanity CLI).",
)
console = Console()


@app.command("list")
def list_adapters() -> None:
    """List the VLA adapters registered with vla_ros2."""

    table = Table(title="vla_ros2 adapters")
    table.add_column("name", style="bold")
    table.add_column("source")
    table.add_column("description")
    for info in list_models():
        table.add_row(info.name, info.source, info.description or "")
    console.print(table)


@app.command("predict")
def predict(
    model: str = typer.Option("dummy", "--model", "-m", help="Adapter name to load."),
    instruction: str = typer.Option(
        "pick up the red block", "--instruction", "-i", help="Task instruction."
    ),
) -> None:
    """Run a single local inference to verify an adapter end-to-end.

    No image is supplied, so this is a wiring check rather than a task run; use
    it to confirm an adapter loads and returns a typed action on this machine.
    """

    try:
        adapter = load_model(model, runtime="local")
        action = adapter.predict(instruction=instruction)
    except VLARos2Error as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]ok[/green] {model} -> {type(action).__name__}")
    console.print(action)


@app.command("compare")
def compare(
    models: list[str] = typer.Option(
        ["dummy", "random", "scripted"],
        "--model",
        "-m",
        help="Adapter names to compare (repeatable).",
    ),
    instruction: str = typer.Option(
        "pick up the red block",
        "--instruction",
        "-i",
        help="Task instruction shared by all adapters.",
    ),
    device: str = typer.Option("auto", "--device", help="Device passed to each adapter."),
) -> None:
    """Compare adapters on the same observation (load + one inference each)."""

    from vla_ros2.core.types import VLAObservation

    observation = VLAObservation(
        instruction=instruction,
        metadata={"source": "vla_ros2_cli_compare"},
    )
    try:
        results = compare_models(models, observation, device=device)
    except VLARos2Error as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="adapter comparison")
    table.add_column("model", style="bold")
    table.add_column("ok")
    table.add_column("load_ms", justify="right")
    table.add_column("infer_ms", justify="right")
    table.add_column("action_preview")
    for item in results:
        load_ms = f"{item.load_ms:.1f}" if item.load_ms is not None else "-"
        infer_ms = f"{item.infer_ms:.1f}" if item.infer_ms is not None else "-"
        preview = item.action_preview or (item.error or "")
        table.add_row(item.name, str(item.ok), load_ms, infer_ms, preview)
    console.print(table)

    if not all(item.ok for item in results):
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
