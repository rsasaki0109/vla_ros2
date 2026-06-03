from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from shlex import quote
from typing import TYPE_CHECKING, Annotated, Any
from zipfile import ZIP_DEFLATED, ZipFile

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
report_app = typer.Typer(help="Package runtime logs and report artifacts.")
gpu_app = typer.Typer(help="Check CUDA runtime paths.")
ros_app = typer.Typer(help="Run ROS2 smoke workflows and reports.")
app.add_typer(demo_app, name="demo")
app.add_typer(compare_app, name="compare")
app.add_typer(report_app, name="report")
app.add_typer(gpu_app, name="gpu")
app.add_typer(ros_app, name="ros")


def _adapter_status(name: str) -> str:
    if name == "dummy":
        return "available"
    if name == "openvla":
        missing = [dep for dep in ("torch", "transformers") if find_spec(dep) is None]
        if missing:
            return 'missing optional deps: pip install "vla_zoo[openvla]"'
        return "available"
    if name == "smolvla":
        missing = [dep for dep in ("torch", "lerobot") if find_spec(dep) is None]
        if missing:
            return 'missing optional deps: pip install "vla_zoo[smolvla]"'
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


def _free_tcp_port(host: str = "127.0.0.1") -> int:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


def _client_host_for_bind_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def _wait_for_http_health(remote_url: str, *, timeout_sec: float) -> str | None:
    deadline = time.monotonic() + timeout_sec
    last_error = "server did not answer before timeout"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{remote_url.rstrip('/')}/health", timeout=0.5) as resp:
                status = resp.getcode()
            if status == 200:
                return None
            last_error = f"health endpoint returned HTTP {status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(0.2)
    return last_error


def _parse_name_list(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise typer.BadParameter("At least one model name is required.")
    return names


def _parse_optional_name_list(value: str | None) -> list[str]:
    if not value:
        return []
    return _parse_name_list(value)


def _model_load_kwargs(
    *,
    pretrained: str | None,
    device: str | None,
    dtype: str | None,
    unnorm_key: str | None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if pretrained:
        kwargs["pretrained"] = pretrained
    if device:
        kwargs["device"] = device
    if dtype:
        kwargs["dtype"] = dtype
    if unnorm_key:
        kwargs["unnorm_key"] = unnorm_key
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    return kwargs


def _parse_paths(value: str) -> list[Path]:
    paths = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not paths:
        raise typer.BadParameter("At least one result JSON path is required.")
    return paths


def _parse_optional_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return _parse_paths(value)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _unlink_if_exists(path: Path) -> None:
    with suppress(FileNotFoundError):
        path.unlink()


def _path_has_content(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except FileNotFoundError:
        return False


def _run_process_for_duration(
    command: list[str],
    *,
    duration_sec: float,
    log_path: Path,
    env: dict[str, str],
) -> tuple[int, bool]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        deadline = time.monotonic() + duration_sec
        completed_early = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                completed_early = True
                break
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        if not completed_early and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return int(process.returncode or 0), completed_early


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

    from vla_zoo.compare.cards import adapter_card_payload

    adapter = get_adapter_info(model)
    typer.echo(
        json.dumps(
            adapter_card_payload(adapter, status=_adapter_status(adapter.name)),
            indent=2,
        )
    )


@app.command()
def doctor(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
    no_ros: Annotated[
        bool,
        typer.Option("--no-ros", help="Skip ros2 and colcon command checks."),
    ] = False,
    no_gpu: Annotated[
        bool,
        typer.Option("--no-gpu", help="Skip nvidia-smi and torch CUDA checks."),
    ] = False,
    remote_url: Annotated[
        str | None,
        typer.Option("--remote-url", help="Check a remote vla-zoo server /health endpoint."),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit non-zero if any error-level check fails."),
    ] = False,
) -> None:
    """Run local environment and runtime readiness checks."""

    from vla_zoo.runtime.doctor import format_doctor_table, run_doctor, summarize_checks

    checks = run_doctor(include_ros=not no_ros, include_gpu=not no_gpu, remote_url=remote_url)
    summary = summarize_checks(checks)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "summary": summary,
                    "checks": [check.to_dict() for check in checks],
                },
                indent=2,
            )
        )
    else:
        typer.echo(format_doctor_table(checks))
    if strict and summary["error"] > 0:
        raise typer.Exit(1)


@app.command()
def predict(
    model: Annotated[str, typer.Option("--model", "-m")] = "dummy",
    instruction: Annotated[str, typer.Option("--instruction", "-i")] = "test",
    pretrained: Annotated[
        str | None,
        typer.Option(
            "--pretrained",
            help="Model checkpoint/repository for adapters such as OpenVLA.",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Device passed to local adapters, for example cuda:0."),
    ] = None,
    dtype: Annotated[
        str | None,
        typer.Option("--dtype", help="Model dtype for local adapters, for example bfloat16."),
    ] = None,
    unnorm_key: Annotated[
        str | None,
        typer.Option("--unnorm-key", help="Dataset/action unnormalization key for adapters."),
    ] = None,
) -> None:
    """Run one prediction and print JSON."""

    try:
        loaded = load_model(
            model,
            **_model_load_kwargs(
                pretrained=pretrained,
                device=device,
                dtype=dtype,
                unnorm_key=unnorm_key,
            ),
        )
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


@app.command("serve-plan")
def serve_plan(
    models: Annotated[
        str,
        typer.Option(
            "--models",
            help="Comma-separated model servers to plan.",
        ),
    ] = "openvla,pi0,smolvla,groot",
    host: Annotated[str, typer.Option("--host", help="Bind host for each GPU server.")] = "0.0.0.0",
    public_host: Annotated[
        str,
        typer.Option("--public-host", help="Hostname/IP reachable from the robot-side runtime."),
    ] = "gpu-box",
    base_port: Annotated[
        int,
        typer.Option("--base-port", help="First port; later models increment by one."),
    ] = 8001,
    device: Annotated[
        str,
        typer.Option("--device", help="Device passed to local GPU adapters."),
    ] = "cuda:0",
    dtype: Annotated[
        str | None,
        typer.Option("--dtype", help="Optional dtype for adapters that expose dtype."),
    ] = None,
    unnorm_key: Annotated[
        str | None,
        typer.Option("--unnorm-key", help="OpenVLA dataset/action unnormalization key."),
    ] = "bridge_orig",
    pretrained_map: Annotated[
        str | None,
        typer.Option(
            "--pretrained-map",
            help=(
                "Comma-separated model=checkpoint overrides, for example "
                "pi0=lerobot/pi0_base,smolvla=lerobot/smolvla_base."
            ),
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write JSON server plan."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write Markdown server plan."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print JSON instead of Markdown."),
    ] = False,
) -> None:
    """Generate a remote GPU server plan for heavyweight VLA comparisons."""

    from vla_zoo.runtime.server_plan import build_server_plan, format_server_plan_markdown

    plan = build_server_plan(
        _parse_name_list(models),
        host=host,
        public_host=public_host,
        base_port=base_port,
        device=device,
        dtype=dtype,
        unnorm_key=unnorm_key,
        pretrained=_parse_remote_map(pretrained_map),
    )
    payload = plan.to_dict()
    markdown = format_server_plan_markdown(plan)
    if out is not None:
        _write_text(out, json.dumps(payload, indent=2) + "\n")
    if markdown_out is not None:
        _write_text(markdown_out, markdown)
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(markdown)


@app.command("smolvla-remote-plan")
def smolvla_remote_plan(
    pretrained: Annotated[
        str,
        typer.Option("--pretrained", help="SmolVLA checkpoint/repository to serve."),
    ] = "lerobot/smolvla_base",
    host: Annotated[
        str,
        typer.Option("--host", help="Bind host for the SmolVLA server."),
    ] = "0.0.0.0",
    public_host: Annotated[
        str,
        typer.Option("--public-host", help="Hostname/IP reachable from the robot-side runtime."),
    ] = "gpu-box",
    port: Annotated[int, typer.Option("--port", help="Server port.")] = 8000,
    device: Annotated[
        str,
        typer.Option("--device", help="Device passed to the SmolVLA adapter."),
    ] = "cuda:0",
    dtype: Annotated[
        str | None,
        typer.Option("--dtype", help="Optional model dtype, for example bfloat16."),
    ] = None,
    venv_dir: Annotated[
        str,
        typer.Option("--venv-dir", help="Isolated virtual environment directory for the server."),
    ] = ".venv-smolvla",
    instruction: Annotated[
        str,
        typer.Option("--instruction", help="Instruction used by the robot-side probe."),
    ] = "pick up the red block",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write JSON plan."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write Markdown plan."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print JSON instead of Markdown."),
    ] = False,
) -> None:
    """Generate an isolated-environment remote SmolVLA serving plan."""

    from vla_zoo.runtime.smolvla_plan import (
        build_smolvla_remote_plan,
        format_smolvla_remote_plan_markdown,
    )

    plan = build_smolvla_remote_plan(
        pretrained=pretrained,
        host=host,
        public_host=public_host,
        port=port,
        device=device,
        dtype=dtype,
        venv_dir=venv_dir,
        instruction=instruction,
    )
    payload = plan.to_dict()
    markdown = format_smolvla_remote_plan_markdown(plan)
    if out is not None:
        _write_text(out, json.dumps(payload, indent=2) + "\n")
    if markdown_out is not None:
        _write_text(markdown_out, markdown)
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(markdown)


@app.command()
def serve(
    model: Annotated[str, typer.Option("--model", "-m")] = "dummy",
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port")] = 8000,
    pretrained: Annotated[
        str | None,
        typer.Option(
            "--pretrained",
            help="Model checkpoint/repository for adapters such as OpenVLA.",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Device passed to local adapters, for example cuda:0."),
    ] = None,
    dtype: Annotated[
        str | None,
        typer.Option("--dtype", help="Model dtype for local adapters, for example bfloat16."),
    ] = None,
    unnorm_key: Annotated[
        str | None,
        typer.Option("--unnorm-key", help="Dataset/action unnormalization key for adapters."),
    ] = None,
    load_in_4bit: Annotated[
        bool,
        typer.Option(
            "--load-in-4bit",
            help="Load with 4-bit (nf4) quantization; fits OpenVLA-7b on a 16 GB GPU.",
        ),
    ] = False,
    load_in_8bit: Annotated[
        bool,
        typer.Option("--load-in-8bit", help="Load with 8-bit quantization (bitsandbytes)."),
    ] = False,
) -> None:
    """Start the optional FastAPI inference server."""

    from vla_zoo.runtime.server import run_server

    try:
        run_server(
            model_name=model,
            host=host,
            port=port,
            **_model_load_kwargs(
                pretrained=pretrained,
                device=device,
                dtype=dtype,
                unnorm_key=unnorm_key,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            ),
        )
    except MissingDependencyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command("remote-probe")
def remote_probe(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model name the remote server hosts."),
    ] = "openvla",
    remote_url: Annotated[
        str,
        typer.Option("--remote-url", help="Base URL of the server, e.g. http://gpu-box:8000."),
    ] = "http://gpu-box:8000",
    instruction: Annotated[
        str,
        typer.Option("--instruction", "-i", help="Instruction sent in the probe request."),
    ] = "pick up the red block",
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="HTTP timeout in seconds for health and predict."),
    ] = 30.0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the recorded probe result JSON."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write a Markdown probe report."),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit non-zero if the probe did not complete."),
    ] = False,
) -> None:
    """Check a remote server's health, then record one /v1/predict response."""

    from vla_zoo.runtime.remote_probe import format_remote_probe_markdown, probe_remote_model

    result = probe_remote_model(
        model_name=model,
        remote_url=remote_url,
        instruction=instruction,
        timeout=timeout,
    )
    payload = result.to_dict()
    if out is not None:
        _write_text(out, json.dumps(payload, indent=2) + "\n")
    if markdown_out is not None:
        _write_text(markdown_out, format_remote_probe_markdown(result))
    typer.echo(json.dumps(payload, indent=2))
    if strict and not result.is_ok:
        raise typer.Exit(1)


@gpu_app.command("smoke")
def gpu_smoke(
    device: Annotated[
        str,
        typer.Option("--device", help="CUDA device to exercise."),
    ] = "cuda:0",
    dtype: Annotated[
        str,
        typer.Option("--dtype", help="Tensor dtype: float16, bfloat16, or float32."),
    ] = "float16",
    matrix_size: Annotated[
        int,
        typer.Option("--matrix-size", help="Square matrix size for the CUDA matmul."),
    ] = 512,
    iterations: Annotated[
        int,
        typer.Option("--iterations", help="Number of matmul iterations to time."),
    ] = 8,
) -> None:
    """Run a small torch CUDA matmul and print JSON."""

    from vla_zoo.runtime.gpu import run_cuda_smoke

    try:
        result = run_cuda_smoke(
            device=device,
            dtype=dtype,
            matrix_size=matrix_size,
            iterations=iterations,
        )
    except VLAZooError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command()
def bench(
    model: Annotated[str, typer.Option("--model", "-m")] = "dummy",
    benchmark: Annotated[str, typer.Option("--benchmark", "-b")] = "smoke",
    episodes: Annotated[int, typer.Option("--episodes", "-e")] = 3,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    jsonl_out: Annotated[
        Path | None,
        typer.Option("--jsonl-out", help="Write per-episode results as versioned JSONL."),
    ] = None,
    summary_out: Annotated[
        Path | None,
        typer.Option("--summary-out", help="Write the latency/action-rate summary as JSON."),
    ] = None,
    summary_md: Annotated[
        Path | None,
        typer.Option("--summary-md", help="Write the latency/action-rate summary as Markdown."),
    ] = None,
) -> None:
    """Run a benchmark and optionally emit the versioned JSONL result schema."""

    if benchmark not in ("smoke", "libero", "simpler"):
        raise typer.BadParameter(
            "benchmark must be one of: smoke, libero, simpler "
            "(libero/simpler are dependency-gated smoke runners)."
        )
    loaded = load_model(model)

    if benchmark == "smoke" and jsonl_out is None and summary_out is None and summary_md is None:
        typer.echo(json.dumps(run_smoke_benchmark(loaded, episodes=episodes, seed=seed), indent=2))
        return

    from vla_zoo.benchmark.results import (
        format_benchmark_summary_markdown,
        summarize_records,
        write_episode_jsonl,
    )
    from vla_zoo.benchmark.runner import run_smoke_episode_records

    try:
        if benchmark == "libero":
            from vla_zoo.benchmark.libero import run_libero_smoke

            records, action_rate_hz = run_libero_smoke(
                loaded, model_name=model, episodes=episodes
            )
        elif benchmark == "simpler":
            from vla_zoo.benchmark.simpler import run_simpler_smoke

            records, action_rate_hz = run_simpler_smoke(
                loaded, model_name=model, episodes=episodes
            )
        else:
            records, action_rate_hz = run_smoke_episode_records(
                loaded, model_name=model, episodes=episodes, seed=seed
            )
    except (MissingDependencyError, NotImplementedError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    summary = summarize_records(records, action_rate_hz=action_rate_hz)
    if jsonl_out is not None:
        write_episode_jsonl(jsonl_out, records)
        typer.echo(f"JSONL written to {jsonl_out}")
    if summary_out is not None:
        _write_text(summary_out, json.dumps(summary.to_dict(), indent=2) + "\n")
        typer.echo(f"Summary JSON written to {summary_out}")
    if summary_md is not None:
        _write_text(summary_md, format_benchmark_summary_markdown(summary))
        typer.echo(f"Summary Markdown written to {summary_md}")
    typer.echo(json.dumps(summary.to_dict(), indent=2))


@app.command("bench-replay")
def bench_replay(
    action_log: Annotated[
        Path,
        typer.Option("--action-log", help="Path to a recorded vla_actions.jsonl action log."),
    ],
    jsonl_out: Annotated[
        Path | None,
        typer.Option("--jsonl-out", help="Write replay results as versioned JSONL."),
    ] = None,
    summary_out: Annotated[
        Path | None,
        typer.Option("--summary-out", help="Write the latency/action-rate summary as JSON."),
    ] = None,
    summary_md: Annotated[
        Path | None,
        typer.Option("--summary-md", help="Write the latency/action-rate summary as Markdown."),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            help="Override the summary source label (default: the ROS action-replay stub).",
        ),
    ] = None,
    note: Annotated[
        str | None,
        typer.Option("--note", help="Override the summary note (default: the replay-stub note)."),
    ] = None,
) -> None:
    """Replay a recorded JSONL action log into the versioned benchmark result schema.

    This is a ROS bag replay stub: it consumes vla_zoo's own JSONL action logs (native
    rosbag2 .db3/.mcap decoding is future work) and makes no task-success claim. Use
    ``--source`` / ``--note`` when replaying logs from another recorder (e.g. a PyBullet
    action probe) so the summary is labeled honestly.
    """

    from vla_zoo.benchmark.replay import (
        REPLAY_SOURCE,
        ROSBAG_REPLAY_NOTE,
        frames_to_records,
        load_action_log,
        replay_action_rate_hz,
    )
    from vla_zoo.benchmark.results import (
        format_benchmark_summary_markdown,
        summarize_records,
        write_episode_jsonl,
    )

    if not action_log.is_file():
        typer.echo(f"action log not found: {action_log}", err=True)
        raise typer.Exit(1)

    frames = load_action_log(action_log)
    records = frames_to_records(frames, source=source or REPLAY_SOURCE)
    summary = summarize_records(
        records,
        action_rate_hz=replay_action_rate_hz(frames),
        note=note or ROSBAG_REPLAY_NOTE,
    )
    if jsonl_out is not None:
        write_episode_jsonl(jsonl_out, records)
        typer.echo(f"JSONL written to {jsonl_out}")
    if summary_out is not None:
        _write_text(summary_out, json.dumps(summary.to_dict(), indent=2) + "\n")
        typer.echo(f"Summary JSON written to {summary_out}")
    if summary_md is not None:
        _write_text(
            summary_md,
            format_benchmark_summary_markdown(
                summary, title="ROS2 Action Replay Latency / Action-Rate Summary"
            ),
        )
        typer.echo(f"Summary Markdown written to {summary_md}")
    typer.echo(json.dumps(summary.to_dict(), indent=2))


@app.command("bench-report")
def bench_report(
    summaries: Annotated[
        str,
        typer.Option(
            "--summaries",
            help="Comma-separated benchmark summary JSON files (vla-zoo-benchmark/v1).",
        ),
    ],
    html_out: Annotated[
        Path | None,
        typer.Option("--html-out", help="Write the standalone HTML comparison report."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write the Markdown comparison report."),
    ] = None,
    title: Annotated[
        str,
        typer.Option("--title", help="Report title."),
    ] = "Benchmark Comparison",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON instead of a table."),
    ] = False,
) -> None:
    """Render benchmark summaries into a comparison report (HTML + Markdown)."""

    from vla_zoo.benchmark.report import (
        format_benchmark_report_html,
        format_benchmark_report_markdown,
    )
    from vla_zoo.benchmark.results import read_summary_json

    loaded = []
    for raw in _parse_name_list(summaries):
        path = Path(raw)
        if not path.is_file():
            typer.echo(f"summary not found: {path}", err=True)
            raise typer.Exit(1)
        loaded.append(read_summary_json(path))

    if not loaded:
        typer.echo("no summaries provided", err=True)
        raise typer.Exit(1)

    if html_out is not None:
        _write_text(html_out, format_benchmark_report_html(loaded, title=title))
        typer.echo(f"HTML written to {html_out}")
    if markdown_out is not None:
        _write_text(markdown_out, format_benchmark_report_markdown(loaded, title=title))
        typer.echo(f"Markdown written to {markdown_out}")
    if json_output:
        typer.echo(json.dumps([summary.to_dict() for summary in loaded], indent=2))
    else:
        typer.echo(f"{'model':<12} {'source':<22} {'p50 ms':>10} {'rate Hz':>10}")
        typer.echo(f"{'-' * 12} {'-' * 22} {'-' * 10} {'-' * 10}")
        for summary in loaded:
            p50 = "-" if summary.latency_ms_p50 is None else f"{summary.latency_ms_p50:.2f}"
            rate = "-" if summary.action_rate_hz is None else f"{summary.action_rate_hz:.2f}"
            typer.echo(f"{summary.model:<12} {summary.source:<22} {p50:>10} {rate:>10}")


@app.command("bench-aggregate")
def bench_aggregate(
    summaries: Annotated[
        str | None,
        typer.Option(
            "--summaries",
            help="Comma-separated benchmark summary JSON files (vla-zoo-benchmark/v1).",
        ),
    ] = None,
    from_log: Annotated[
        str | None,
        typer.Option(
            "--from-log",
            help=(
                "Comma-separated vla_actions.jsonl action logs to replay into summaries "
                "first (success_rate stays None). Combined with any --summaries inputs."
            ),
        ),
    ] = None,
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            help="Ranking metric: latency_ms_p50/p95/mean (lower better) or action_rate_hz.",
        ),
    ] = "latency_ms_p50",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the machine-readable aggregate JSON."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write the ranked Markdown table."),
    ] = None,
    html_out: Annotated[
        Path | None,
        typer.Option("--html-out", help="Write the standalone HTML ranked aggregate."),
    ] = None,
    title: Annotated[
        str,
        typer.Option("--title", help="Report title."),
    ] = "Benchmark Aggregate (Ranked)",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON instead of a table."),
    ] = False,
) -> None:
    """Merge several benchmark summaries into one table ranked by a runtime metric.

    Inputs can be pre-computed ``vla-zoo-benchmark/v1`` summary JSONs (``--summaries``)
    and/or raw ``vla_actions.jsonl`` action logs replayed on the fly (``--from-log``); the
    two sources are combined before ranking. Replayed logs keep ``success_rate`` ``None``.
    """

    from vla_zoo.benchmark.aggregate import (
        format_aggregate_html,
        format_aggregate_markdown,
        rank_summaries,
    )
    from vla_zoo.benchmark.replay import summarize_action_log
    from vla_zoo.benchmark.results import read_summary_json

    loaded = []
    for raw in _parse_optional_name_list(summaries):
        path = Path(raw)
        if not path.is_file():
            typer.echo(f"summary not found: {path}", err=True)
            raise typer.Exit(1)
        loaded.append(read_summary_json(path))

    for raw in _parse_optional_name_list(from_log):
        path = Path(raw)
        if not path.is_file():
            typer.echo(f"action log not found: {path}", err=True)
            raise typer.Exit(1)
        loaded.append(summarize_action_log(path))

    if not loaded:
        typer.echo("no inputs provided (use --summaries and/or --from-log)", err=True)
        raise typer.Exit(1)

    try:
        report = rank_summaries(loaded, metric=metric)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if out is not None:
        _write_text(out, json.dumps(report.to_dict(), indent=2) + "\n")
        typer.echo(f"JSON written to {out}")
    if markdown_out is not None:
        _write_text(markdown_out, format_aggregate_markdown(report, title=title))
        typer.echo(f"Markdown written to {markdown_out}")
    if html_out is not None:
        _write_text(html_out, format_aggregate_html(report, title=title))
        typer.echo(f"HTML written to {html_out}")
    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        typer.echo(f"ranked by {report.metric} ({report.count} entries)")
        typer.echo(f"{'rank':>4} {'model':<12} {'source':<22} {'value':>12}")
        typer.echo(f"{'-' * 4} {'-' * 12} {'-' * 22} {'-' * 12}")
        for entry in report.ranked:
            rank = "-" if entry.rank is None else str(entry.rank)
            value = "-" if entry.metric_value is None else f"{entry.metric_value:.2f}"
            typer.echo(
                f"{rank:>4} {entry.summary.model:<12} {entry.summary.source:<22} {value:>12}"
            )


def _load_ros_diagnostics(path: Path, status_name: str) -> list[Any]:
    """Reconstruct native diagnostics records from a recorded ROS2 /diagnostics JSONL."""

    from vla_zoo.runtime.diagnostics import diagnostics_from_key_values

    records: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        for status in payload.get("status", []):
            if status_name and status.get("name") != status_name:
                continue
            pairs = {item["key"]: item["value"] for item in status.get("values", [])}
            if pairs:
                records.append(diagnostics_from_key_values(pairs))
    return records


@app.command("diag-report")
def diag_report(
    log: Annotated[
        Path | None,
        typer.Option("--log", help="Native vla-zoo-diagnostics/v1 JSONL log path."),
    ] = None,
    from_ros_log: Annotated[
        Path | None,
        typer.Option(
            "--from-ros-log",
            help="Recorded ROS2 /diagnostics DiagnosticArray JSONL to reconstruct from.",
        ),
    ] = None,
    status_name: Annotated[
        str,
        typer.Option(
            "--status-name",
            help="DiagnosticStatus name to select when reading --from-ros-log.",
        ),
    ] = "vla_zoo/vla_runtime_node",
    jsonl_out: Annotated[
        Path | None,
        typer.Option(
            "--jsonl-out",
            help="Persist the (reconstructed) records as a native vla-zoo-diagnostics/v1 JSONL.",
        ),
    ] = None,
    summary: Annotated[
        bool,
        typer.Option(
            "--summary",
            help="Aggregate the whole log into a time-series summary instead of one snapshot.",
        ),
    ] = False,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write the Markdown snapshot/summary."),
    ] = None,
    title: Annotated[
        str | None,
        typer.Option("--title", help="Report title."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print machine-readable JSON (all records, or the summary with --summary).",
        ),
    ] = False,
) -> None:
    """Render a runtime diagnostics snapshot or summary from a JSONL log.

    The log may be a native vla-zoo-diagnostics/v1 file (--log) or a recorded ROS2
    /diagnostics DiagnosticArray file (--from-ros-log). By default the latest record's
    snapshot is rendered; --summary aggregates the whole log into a time-series view.
    """

    from vla_zoo.runtime.diagnostics import (
        format_diagnostics_markdown,
        format_diagnostics_summary_markdown,
        read_diagnostics_jsonl,
        summarize_diagnostics,
        write_diagnostics_jsonl,
    )

    if (log is None) == (from_ros_log is None):
        typer.echo("Provide exactly one of --log or --from-ros-log.", err=True)
        raise typer.Exit(1)

    try:
        if log is not None:
            if not log.is_file():
                typer.echo(f"log not found: {log}", err=True)
                raise typer.Exit(1)
            records = read_diagnostics_jsonl(log)
        else:
            assert from_ros_log is not None
            if not from_ros_log.is_file():
                typer.echo(f"log not found: {from_ros_log}", err=True)
                raise typer.Exit(1)
            records = _load_ros_diagnostics(from_ros_log, status_name)
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if not records:
        typer.echo("no diagnostics records found", err=True)
        raise typer.Exit(1)

    if jsonl_out is not None:
        write_diagnostics_jsonl(jsonl_out, records)
        typer.echo(f"JSONL written to {jsonl_out}")

    if summary:
        reduced = summarize_diagnostics(records)
        report_title = title or "Runtime Diagnostics Summary"
        rendered = format_diagnostics_summary_markdown(reduced, title=report_title)
        payload: object = reduced.to_dict()
    else:
        latest = records[-1]
        report_title = title or "Runtime Diagnostics Snapshot"
        rendered = format_diagnostics_markdown(latest, title=report_title)
        payload = [record.to_dict() for record in records]

    if markdown_out is not None:
        _write_text(markdown_out, rendered)
        typer.echo(f"Markdown written to {markdown_out}")
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    elif markdown_out is None and jsonl_out is None:
        typer.echo(rendered)
    else:
        typer.echo(f"{len(records)} record(s) processed")


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


@compare_app.command("methods")
def compare_methods(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write JSON profiles to this path."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write a README-ready Markdown table."),
    ] = None,
) -> None:
    """Compare VLA method profiles without loading model weights."""

    from vla_zoo.compare.profiles import format_method_profiles_markdown, method_profiles

    profiles = method_profiles(status_provider=_adapter_status)
    payload = json.dumps([profile.to_dict() for profile in profiles], indent=2)
    if out is not None:
        _write_text(out, f"{payload}\n")
    if markdown_out is not None:
        _write_text(markdown_out, format_method_profiles_markdown(profiles))
    if json_output:
        typer.echo(payload)
    else:
        typer.echo(
            f"{'method':<11} {'family':<28} {'action':<22} {'chunks':<14} "
            f"{'remote':<24} status"
        )
        typer.echo(
            f"{'-' * 11} {'-' * 28} {'-' * 22} {'-' * 14} "
            f"{'-' * 24} {'-' * 32}"
        )
        for profile in profiles:
            action = f"{profile.action_space} {profile.action_shape}"
            typer.echo(
                f"{profile.name:<11} "
                f"{_shorten(profile.family, 28):<28} "
                f"{_shorten(action, 22):<22} "
                f"{_shorten(profile.action_chunks, 14):<14} "
                f"{_shorten(profile.remote_runtime, 24):<24} "
                f"{_shorten(profile.status, 48)}"
            )
    if out is not None:
        typer.echo(f"JSON written to {out}")
    if markdown_out is not None:
        typer.echo(f"Markdown written to {markdown_out}")


@compare_app.command("evidence")
def compare_evidence(
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated adapters to include in the evidence matrix.",
        ),
    ] = "dummy,scripted,random,openvla,pi0,smolvla,groot",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write JSON evidence matrix."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write Markdown evidence matrix."),
    ] = None,
    html_out: Annotated[
        Path | None,
        typer.Option("--html-out", help="Write standalone HTML evidence matrix."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
    title: Annotated[
        str,
        typer.Option("--title", help="Report title."),
    ] = "VLA Model Evidence Matrix",
) -> None:
    """Report which VLA runtime paths have checked-in evidence."""

    from vla_zoo.compare.evidence import (
        build_evidence_matrix,
        evidence_matrix_payload,
        format_evidence_matrix_html,
        format_evidence_matrix_markdown,
    )

    records = build_evidence_matrix(
        _parse_name_list(models),
        status_provider=_adapter_status,
    )
    payload = evidence_matrix_payload(records)
    json_payload = json.dumps(payload, indent=2)
    markdown = format_evidence_matrix_markdown(records, title=title)
    html = format_evidence_matrix_html(records, title=title)
    if out is not None:
        _write_text(out, f"{json_payload}\n")
    if markdown_out is not None:
        _write_text(markdown_out, markdown)
    if html_out is not None:
        _write_text(html_out, html)
    if json_output:
        typer.echo(json_payload)
    else:
        typer.echo(f"{'model':<11} {'contract':<10} {'gpu':<14} {'remote':<14} next step")
        typer.echo(f"{'-' * 11} {'-' * 10} {'-' * 14} {'-' * 14} {'-' * 42}")
        for record in records:
            cells = record.evidence
            typer.echo(
                f"{record.model:<11} "
                f"{cells['contract'].status:<10} "
                f"{cells['gpu_inference'].status:<14} "
                f"{cells['remote_server'].status:<14} "
                f"{_shorten(record.next_step, 72)}"
            )
    if out is not None:
        typer.echo(f"JSON written to {out}")
    if markdown_out is not None:
        typer.echo(f"Markdown written to {markdown_out}")
    if html_out is not None:
        typer.echo(f"HTML written to {html_out}")


@compare_app.command("cards")
def compare_cards(
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated adapters to generate cards for.",
        ),
    ] = "dummy,scripted,random,openvla,pi0,smolvla,groot",
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", "-o", help="Directory for generated Markdown cards."),
    ] = Path("docs/adapters"),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print generated paths as JSON."),
    ] = False,
) -> None:
    """Generate adapter capability cards from registry metadata."""

    from vla_zoo.compare.cards import write_adapter_cards

    paths = write_adapter_cards(
        out_dir,
        models=_parse_name_list(models),
        status_provider=_adapter_status,
    )
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "files": [str(path) for path in paths],
                },
                indent=2,
            )
        )
        return
    typer.echo(f"Adapter cards written to {out_dir}")
    for path in paths:
        typer.echo(f"- {path}")


@compare_app.command("compatibility")
def compare_compatibility(
    robot_profile: Annotated[
        str,
        typer.Option(
            "--robot-profile",
            "-r",
            help="Built-in robot profile to check against.",
        ),
    ] = "single-camera-eef",
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated adapters to check.",
        ),
    ] = "dummy,scripted,random,openvla,pi0,smolvla,groot",
    list_profiles: Annotated[
        bool,
        typer.Option("--list-profiles", help="List built-in robot profiles and exit."),
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write JSON compatibility report."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write Markdown compatibility report."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Check adapter contracts against robot-side capability profiles."""

    from vla_zoo.compare.compatibility import (
        ROBOT_PROFILE_PRESETS,
        compatibility_matrix,
        format_compatibility_markdown,
        format_robot_profiles_markdown,
        get_robot_profile,
    )

    if list_profiles:
        typer.echo(format_robot_profiles_markdown())
        return
    try:
        robot = get_robot_profile(robot_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    results = compatibility_matrix(
        robot=robot,
        models=_parse_name_list(models),
        status_provider=_adapter_status,
    )
    payload = {
        "robot_profile": robot.to_dict(),
        "available_profiles": sorted(ROBOT_PROFILE_PRESETS),
        "results": [result.to_dict() for result in results],
    }
    json_payload = json.dumps(payload, indent=2)
    markdown = format_compatibility_markdown(results, robot=robot)
    if out is not None:
        _write_text(out, f"{json_payload}\n")
    if markdown_out is not None:
        _write_text(markdown_out, markdown)
    if json_output:
        typer.echo(json_payload)
    else:
        typer.echo(
            f"{'model':<11} {'fit':<14} {'score':<5} {'adapter':<18} "
            f"{'action':<24} first issue"
        )
        typer.echo(
            f"{'-' * 11} {'-' * 14} {'-' * 5} {'-' * 18} {'-' * 24} {'-' * 40}"
        )
        for result in results:
            first_issue = result.issues[0].message if result.issues else "-"
            action = f"{result.action_space} {result.action_shape}"
            typer.echo(
                f"{result.model:<11} "
                f"{result.status:<14} "
                f"{result.score:<5} "
                f"{_shorten(result.adapter_status, 18):<18} "
                f"{_shorten(action, 24):<24} "
                f"{_shorten(first_issue, 64)}"
            )
    if out is not None:
        typer.echo(f"JSON written to {out}")
    if markdown_out is not None:
        typer.echo(f"Markdown written to {markdown_out}")


@compare_app.command("suite")
def compare_suite(
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", "-o", help="Output directory for suite artifacts."),
    ] = Path("results/vla_compare_suite"),
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated adapters for the optional PyBullet smoke comparison.",
        ),
    ] = "dummy,scripted,random",
    runtime: Annotated[str, typer.Option("--runtime")] = "local",
    remote_url: Annotated[str, typer.Option("--remote-url")] = "http://localhost:8000",
    remote_map: Annotated[
        str | None,
        typer.Option(
            "--remote-map",
            help="Comma-separated model=url overrides for remote PyBullet comparisons.",
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
    pybullet: Annotated[
        bool,
        typer.Option("--pybullet/--no-pybullet", help="Run the PyBullet smoke comparison."),
    ] = True,
    dashboard: Annotated[
        bool,
        typer.Option("--dashboard/--no-dashboard", help="Generate dashboard HTML."),
    ] = True,
) -> None:
    """Generate a shareable VLA comparison artifact directory."""

    from vla_zoo.compare.profiles import format_method_profiles_markdown, method_profiles
    from vla_zoo.compare.suite import SuiteArtifact, format_suite_readme

    out_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    command_parts = [
        "vla-zoo",
        "compare",
        "suite",
        "--out-dir",
        str(out_dir),
        "--models",
        models,
        "--runtime",
        runtime,
        "--instruction",
        instruction,
        "--model-call-every",
        str(model_call_every),
        "--render-stride",
        str(render_stride),
    ]
    if not pybullet:
        command_parts.append("--no-pybullet")
    if runtime == "remote":
        command_parts.extend(["--remote-url", remote_url])
    if remote_map:
        command_parts.extend(["--remote-map", remote_map])
    if allow_local_heavy:
        command_parts.append("--allow-local-heavy")
    if not dashboard:
        command_parts.append("--no-dashboard")
    command = " ".join(quote(part) for part in command_parts)

    profiles = method_profiles(status_provider=_adapter_status)
    method_profiles_json = out_dir / "method_profiles.json"
    method_profiles_md = out_dir / "method_profiles.md"
    method_markdown = format_method_profiles_markdown(profiles)
    _write_text(
        method_profiles_json,
        json.dumps([profile.to_dict() for profile in profiles], indent=2) + "\n",
    )
    _write_text(method_profiles_md, method_markdown)
    artifacts = [
        SuiteArtifact(
            label="method profiles JSON",
            path=method_profiles_json.name,
            description="structured adapter integration profiles",
        ),
        SuiteArtifact(
            label="method profiles Markdown",
            path=method_profiles_md.name,
            description="README-ready adapter integration table",
        ),
    ]

    pybullet_markdown: str | None = None
    if pybullet:
        from vla_zoo.demo.pybullet import (
            compare_pybullet_models,
            format_pybullet_comparison_html,
            format_pybullet_comparison_markdown,
        )

        model_names = [item.strip() for item in models.split(",") if item.strip()]
        if not model_names:
            raise typer.BadParameter("At least one model name is required.")
        results = compare_pybullet_models(
            model_names,
            runtime=runtime,
            remote_url=remote_url,
            remote_urls=_parse_remote_map(remote_map) or None,
            instruction=instruction,
            model_call_every=model_call_every,
            render_stride=render_stride,
            allow_local_heavy=allow_local_heavy,
        )
        pybullet_json = out_dir / "pybullet_results.json"
        pybullet_md = out_dir / "pybullet_results.md"
        pybullet_html = out_dir / "pybullet_report.html"
        pybullet_markdown = format_pybullet_comparison_markdown(results)
        pybullet_payload = json.dumps(
            [asdict(result) for result in results],
            indent=2,
        )
        _write_text(pybullet_json, f"{pybullet_payload}\n")
        _write_text(pybullet_md, pybullet_markdown)
        _write_text(pybullet_html, format_pybullet_comparison_html(results))
        artifacts.extend(
            [
                SuiteArtifact(
                    label="PyBullet JSON",
                    path=pybullet_json.name,
                    description="deterministic smoke-scene runtime and task telemetry",
                ),
                SuiteArtifact(
                    label="PyBullet Markdown",
                    path=pybullet_md.name,
                    description="README-ready PyBullet comparison table",
                ),
                SuiteArtifact(
                    label="PyBullet HTML",
                    path=pybullet_html.name,
                    description="self-contained PyBullet comparison report",
                ),
            ]
        )
        if dashboard:
            from vla_zoo.runtime.dashboard import (
                dashboard_records_from_payload,
                format_comparison_dashboard_html,
            )

            dashboard_html = out_dir / "runtime_dashboard.html"
            records = dashboard_records_from_payload([asdict(result) for result in results])
            _write_text(
                dashboard_html,
                format_comparison_dashboard_html(records, title="vla_zoo Comparison Suite"),
            )
            artifacts.append(
                SuiteArtifact(
                    label="runtime dashboard",
                    path=dashboard_html.name,
                    description="interactive static dashboard for PyBullet results",
                )
            )

    readme_path = out_dir / "README.md"
    _write_text(
        readme_path,
        format_suite_readme(
            title="vla_zoo Comparison Suite",
            created_at=created_at,
            command=command,
            artifacts=artifacts,
            method_profiles_markdown=method_markdown,
            pybullet_markdown=pybullet_markdown,
        ),
    )
    typer.echo(f"Comparison suite written to {out_dir}")
    for artifact in [*artifacts, SuiteArtifact("suite README", readme_path.name, "index")]:
        typer.echo(f"- {artifact.path}: {artifact.description}")


@compare_app.command("dashboard")
def compare_dashboard(
    results: Annotated[
        str | None,
        typer.Option(
            "--results",
            "-r",
            help="Comma-separated comparison JSON or JSONL result paths.",
        ),
    ] = None,
    status_logs: Annotated[
        str | None,
        typer.Option(
            "--status-log",
            help="Comma-separated ROS2 VLAStatus JSON or JSONL paths.",
        ),
    ] = None,
    diagnostics_logs: Annotated[
        str | None,
        typer.Option(
            "--diagnostics-log",
            help="Comma-separated DiagnosticArray/DiagnosticStatus JSON or JSONL paths.",
        ),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output HTML dashboard path."),
    ] = Path("results/vla_runtime_dashboard.html"),
    title: Annotated[
        str,
        typer.Option("--title", help="Dashboard title."),
    ] = "vla_zoo Runtime Dashboard",
) -> None:
    """Build an interactive static dashboard from comparison or ROS2 runtime logs."""

    from vla_zoo.runtime.dashboard import (
        format_comparison_dashboard_html,
        load_dashboard_records,
        load_diagnostics_summaries,
        load_runtime_dashboard_records,
    )

    diagnostics_paths = _parse_optional_paths(diagnostics_logs)
    try:
        records = load_dashboard_records(_parse_optional_paths(results))
        runtime_log_paths = [
            *_parse_optional_paths(status_logs),
            *diagnostics_paths,
        ]
        records.extend(load_runtime_dashboard_records(runtime_log_paths))
        diagnostics_summaries = load_diagnostics_summaries(diagnostics_paths)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if not records:
        typer.echo(
            "At least one --results, --status-log, or --diagnostics-log path is required.",
            err=True,
        )
        raise typer.Exit(1)
    _write_text(
        out,
        format_comparison_dashboard_html(
            records, title=title, diagnostics_summaries=diagnostics_summaries
        ),
    )
    typer.echo(f"Dashboard written to {out}")


def _adapter_inventory() -> list[dict[str, Any]]:
    adapters: list[dict[str, Any]] = []
    for adapter in list_models():
        adapters.append(
            {
                "name": adapter.name,
                "source": adapter.source,
                "status": _adapter_status(adapter.name),
                "aliases": list(adapter.aliases),
                "experimental": adapter.experimental,
                "domain": adapter.domain,
                "description": adapter.description,
                "install_hint": adapter.install_hint,
            }
        )
    return adapters


def _bundle_arcname(kind: str, path: Path, index: int) -> str:
    return f"inputs/{kind}/{index:02d}_{path.name}"


@report_app.command("bundle")
def report_bundle(
    results: Annotated[
        str | None,
        typer.Option(
            "--results",
            "-r",
            help="Comma-separated comparison JSON or JSONL result paths.",
        ),
    ] = None,
    status_logs: Annotated[
        str | None,
        typer.Option(
            "--status-log",
            help="Comma-separated ROS2 VLAStatus JSON or JSONL paths.",
        ),
    ] = None,
    diagnostics_logs: Annotated[
        str | None,
        typer.Option(
            "--diagnostics-log",
            help="Comma-separated DiagnosticArray/DiagnosticStatus JSON or JSONL paths.",
        ),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output zip bundle path."),
    ] = Path("results/vla_runtime_report_bundle.zip"),
    title: Annotated[
        str,
        typer.Option("--title", help="Dashboard title."),
    ] = "vla_zoo Runtime Report",
) -> None:
    """Package runtime logs, dashboard HTML, and metadata into one zip artifact."""

    from vla_zoo.runtime.dashboard import (
        format_comparison_dashboard_html,
        load_dashboard_records,
        load_diagnostics_summaries,
        load_runtime_dashboard_records,
    )

    result_paths = _parse_optional_paths(results)
    status_paths = _parse_optional_paths(status_logs)
    diagnostics_paths = _parse_optional_paths(diagnostics_logs)
    try:
        records = load_dashboard_records(result_paths)
        records.extend(load_runtime_dashboard_records([*status_paths, *diagnostics_paths]))
        diagnostics_summaries = load_diagnostics_summaries(diagnostics_paths)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if not records:
        typer.echo(
            "At least one --results, --status-log, or --diagnostics-log path is required.",
            err=True,
        )
        raise typer.Exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)
    records_payload = [asdict(record) for record in records]
    dashboard_html = format_comparison_dashboard_html(
        records, title=title, diagnostics_summaries=diagnostics_summaries
    )
    metadata = {
        "schema": "vla_zoo.report_bundle.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "record_count": len(records),
        "inputs": {
            "results": [str(path) for path in result_paths],
            "status_logs": [str(path) for path in status_paths],
            "diagnostics_logs": [str(path) for path in diagnostics_paths],
        },
        "adapters": _adapter_inventory(),
    }
    readme = (
        "vla_zoo runtime report bundle\n\n"
        "Open dashboard.html in a browser first. records.json contains the normalized "
        "dashboard records. metadata.json contains adapter inventory and input paths. "
        "Original inputs are copied under inputs/.\n"
    )

    with ZipFile(out, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.writestr("README.txt", readme)
        bundle.writestr("dashboard.html", dashboard_html)
        bundle.writestr("records.json", json.dumps(records_payload, indent=2) + "\n")
        bundle.writestr("metadata.json", json.dumps(metadata, indent=2) + "\n")
        for index, path in enumerate(result_paths):
            bundle.write(path, _bundle_arcname("results", path, index))
        for index, path in enumerate(status_paths):
            bundle.write(path, _bundle_arcname("status", path, index))
        for index, path in enumerate(diagnostics_paths):
            bundle.write(path, _bundle_arcname("diagnostics", path, index))

    typer.echo(f"Report bundle written to {out}")


@report_app.command("link-check")
def report_link_check(
    paths: Annotated[
        str,
        typer.Option(
            "--paths",
            help="Comma-separated Markdown/HTML files to scan for local links.",
        ),
    ],
    root: Annotated[
        Path,
        typer.Option("--root", help="Repository root for resolving /-rooted links."),
    ] = Path("."),
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write machine-readable JSON report."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print JSON instead of the status table."),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit non-zero if any local link is broken."),
    ] = False,
) -> None:
    """Verify that local links in README/Pages artifacts resolve to existing files."""

    from vla_zoo.docs.links import check_paths, format_link_report_table, link_report_payload

    report = check_paths(_parse_paths(paths), root=root)
    payload = json.dumps(link_report_payload(report), indent=2)
    if out is not None:
        _write_text(out, f"{payload}\n")
    if json_output:
        typer.echo(payload)
    else:
        typer.echo(format_link_report_table(report))
        for broken in report.broken:
            typer.echo(f"broken: {broken.source} -> {broken.link}", err=True)
    if out is not None:
        typer.echo(f"JSON written to {out}")
    if strict and report.broken:
        raise typer.Exit(1)


@report_app.command("index")
def report_index(
    root: Annotated[
        Path,
        typer.Option("--root", help="Repository root for resolving artifact paths."),
    ] = Path("."),
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write JSON artifact index."),
    ] = None,
    html_out: Annotated[
        Path | None,
        typer.Option("--html-out", help="Write standalone HTML artifact index."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print JSON instead of the status table."),
    ] = False,
    title: Annotated[
        str,
        typer.Option("--title", help="Artifact index title."),
    ] = "vla_zoo Artifact Index",
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit non-zero if any catalogued artifact is missing."),
    ] = False,
) -> None:
    """Build a curated, machine-readable index of README/Pages artifacts."""

    from vla_zoo.docs.artifact_index import (
        artifact_index_payload,
        build_artifact_index,
        format_artifact_index_html,
        format_artifact_index_table,
    )

    index = build_artifact_index(root=root)
    payload = json.dumps(artifact_index_payload(index), indent=2)
    if out is not None:
        _write_text(out, f"{payload}\n")
    if html_out is not None:
        _write_text(
            html_out,
            format_artifact_index_html(
                index,
                title=title,
                root=root,
                html_dir=html_out.parent,
            ),
        )
    if json_output:
        typer.echo(payload)
    else:
        typer.echo(format_artifact_index_table(index))
        for entry in index.missing:
            typer.echo(f"missing: {entry.path}", err=True)
    if out is not None:
        typer.echo(f"JSON written to {out}")
    if html_out is not None:
        typer.echo(f"HTML written to {html_out}")
    if strict and index.missing:
        raise typer.Exit(1)


@ros_app.command("action-analyze")
def ros_action_analyze(
    action_log: Annotated[
        Path,
        typer.Option("--action-log", help="ROS2 VLAAction JSONL path."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output JSON analysis path."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Output Markdown analysis path."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
    max_gap_warn_sec: Annotated[
        float,
        typer.Option("--max-gap-warn-sec", help="Warn when action gaps exceed this value."),
    ] = 1.0,
    min_rate_warn_hz: Annotated[
        float,
        typer.Option("--min-rate-warn-hz", help="Warn when action rate is below this value."),
    ] = 1.0,
    title: Annotated[
        str,
        typer.Option("--title", help="Markdown report title."),
    ] = "vla_zoo Action Analysis",
) -> None:
    """Analyze recorded ROS2 VLAAction JSONL for timing and action-quality signals."""

    from vla_zoo.runtime.action_trace import (
        analyze_action_trace,
        format_action_analysis_markdown,
        load_action_trace_events,
    )

    try:
        analysis = analyze_action_trace(
            load_action_trace_events(action_log),
            max_gap_warn_sec=max_gap_warn_sec,
            min_rate_warn_hz=min_rate_warn_hz,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    payload = json.dumps(asdict(analysis), indent=2)
    markdown = format_action_analysis_markdown(analysis, title=title)
    if out is not None:
        _write_text(out, f"{payload}\n")
        typer.echo(f"Action analysis JSON written to {out}")
    if markdown_out is not None:
        _write_text(markdown_out, markdown)
        typer.echo(f"Action analysis Markdown written to {markdown_out}")
    if json_output:
        typer.echo(payload)
    elif out is None and markdown_out is None:
        typer.echo(markdown)


@ros_app.command("action-trace")
def ros_action_trace(
    action_log: Annotated[
        Path,
        typer.Option("--action-log", help="ROS2 VLAAction JSONL path."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output HTML action trace path."),
    ] = Path("results/ros2_smoke/action_trace.html"),
    title: Annotated[
        str,
        typer.Option("--title", help="Action trace title."),
    ] = "vla_zoo Action Trace",
) -> None:
    """Build a static HTML action trace from recorded ROS2 VLAAction JSONL."""

    from vla_zoo.runtime.action_trace import format_action_trace_html, load_action_trace_events

    try:
        events = load_action_trace_events(action_log)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if not events:
        typer.echo(f"No action records found in {action_log}", err=True)
        raise typer.Exit(1)
    _write_text(out, format_action_trace_html(events, title=title))
    typer.echo(f"Action trace written to {out}")


def _finalize_ros_report(
    *,
    output_dir: Path,
    action_log: Path,
    status_log: Path,
    diagnostics_log: Path,
    dashboard_out: Path,
    bundle_out: Path,
    action_trace_out: Path,
    action_analysis_out: Path,
    action_analysis_markdown_out: Path,
    launch_log: Path,
    title: str,
    label: str,
) -> None:
    diagnostics_arg = str(diagnostics_log) if diagnostics_log.exists() else None
    if not _path_has_content(status_log):
        typer.echo(
            f"No status records were written to {status_log}. "
            "Run longer, check ROS2 discovery, or inspect the launch log.",
            err=True,
        )
        if launch_log.exists():
            typer.echo(f"Launch log: {launch_log}", err=True)
        raise typer.Exit(1)
    if diagnostics_log.exists() and not _path_has_content(diagnostics_log):
        typer.echo(f"Warning: diagnostics log is empty: {diagnostics_log}", err=True)

    compare_dashboard(
        results=None,
        status_logs=str(status_log),
        diagnostics_logs=diagnostics_arg,
        out=dashboard_out,
        title=title,
    )
    report_bundle(
        results=None,
        status_logs=str(status_log),
        diagnostics_logs=diagnostics_arg,
        out=bundle_out,
        title=title,
    )
    if _path_has_content(action_log):
        ros_action_trace(action_log=action_log, out=action_trace_out, title=f"{title}: Actions")
        ros_action_analyze(
            action_log=action_log,
            out=action_analysis_out,
            markdown_out=action_analysis_markdown_out,
            json_output=False,
            max_gap_warn_sec=1.0,
            min_rate_warn_hz=1.0,
            title=f"{title}: Action Analysis",
        )
        if bundle_out.exists():
            with ZipFile(bundle_out, "a", compression=ZIP_DEFLATED) as bundle:
                bundle.write(action_log, "inputs/actions/00_vla_actions.jsonl")
                bundle.write(action_trace_out, "action_trace.html")
                bundle.write(action_analysis_out, "action_analysis.json")
                bundle.write(action_analysis_markdown_out, "action_analysis.md")
    else:
        typer.echo(f"Warning: action log is empty or missing: {action_log}", err=True)
    typer.echo(f"{label} written to {output_dir}")


@ros_app.command("remote-smoke-plan")
def ros_remote_smoke_plan(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Remote model name requested by the ROS2 runtime."),
    ] = "openvla",
    remote_url: Annotated[
        str,
        typer.Option("--remote-url", help="Remote vla-zoo server endpoint."),
    ] = "http://gpu-box:8001",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for ROS2 JSONL logs and reports."),
    ] = Path("results/ros2_remote_smoke"),
    duration_sec: Annotated[
        float,
        typer.Option("--duration-sec", help="Suggested recording duration."),
    ] = 30.0,
    dtype: Annotated[
        str | None,
        typer.Option("--dtype", help="Server-side dtype for adapters that expose dtype."),
    ] = "bfloat16",
    instruction: Annotated[
        str,
        typer.Option("--instruction", help="Instruction passed to the smoke input node."),
    ] = "pick up the red block",
    task_id: Annotated[
        str,
        typer.Option("--task-id", help="Typed instruction task_id for the smoke run."),
    ] = "ros2_remote_smoke_pick_red_block",
    publish_actions_in_dry_run: Annotated[
        bool,
        typer.Option(
            "--publish-actions-in-dry-run/--suppress-actions-in-dry-run",
            help="Publish typed action messages while dry_run remains true.",
        ),
    ] = True,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write JSON plan."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write Markdown plan."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print JSON instead of Markdown."),
    ] = False,
) -> None:
    """Generate commands for ROS2 remote runtime smoke recording."""

    from vla_zoo.runtime.ros_plan import (
        build_ros_remote_smoke_plan,
        format_ros_remote_smoke_plan_markdown,
    )

    plan = build_ros_remote_smoke_plan(
        model_name=model,
        remote_url=remote_url,
        output_dir=str(output_dir),
        duration_sec=duration_sec,
        dtype=dtype,
        instruction=instruction,
        task_id=task_id,
        publish_actions_in_dry_run=publish_actions_in_dry_run,
    )
    payload = plan.to_dict()
    markdown = format_ros_remote_smoke_plan_markdown(plan)
    if out is not None:
        _write_text(out, json.dumps(payload, indent=2) + "\n")
    if markdown_out is not None:
        _write_text(markdown_out, markdown)
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(markdown)


@ros_app.command("remote-smoke-check")
def ros_remote_smoke_check(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory containing ROS2 remote JSONL logs."),
    ] = Path("results/ros2_remote_smoke"),
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Expected remote model name."),
    ] = "openvla",
    remote_url: Annotated[
        str,
        typer.Option("--remote-url", help="Expected remote vla-zoo server endpoint."),
    ] = "http://gpu-box:8001",
    action_log_name: Annotated[
        str,
        typer.Option("--action-log-name", help="Action JSONL filename inside output-dir."),
    ] = "vla_actions.jsonl",
    status_log_name: Annotated[
        str,
        typer.Option("--status-log-name", help="Status JSONL filename inside output-dir."),
    ] = "vla_status.jsonl",
    diagnostics_log_name: Annotated[
        str,
        typer.Option(
            "--diagnostics-log-name",
            help="Diagnostics JSONL filename inside output-dir.",
        ),
    ] = "vla_diagnostics.jsonl",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write machine-readable check JSON."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write Markdown check report."),
    ] = None,
    require_actions: Annotated[
        bool,
        typer.Option("--require-actions/--allow-empty-actions", help="Require action records."),
    ] = True,
    require_diagnostics: Annotated[
        bool,
        typer.Option(
            "--require-diagnostics/--allow-empty-diagnostics",
            help="Require diagnostics records.",
        ),
    ] = True,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print JSON instead of Markdown."),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option("--strict/--no-strict", help="Exit non-zero when checks fail."),
    ] = True,
    title: Annotated[
        str,
        typer.Option("--title", help="Markdown report title."),
    ] = "ROS2 Remote Runtime Smoke Check",
) -> None:
    """Validate ROS2 remote smoke logs for remote runtime evidence."""

    from vla_zoo.runtime.ros_smoke import (
        check_ros_remote_smoke,
        format_ros_remote_smoke_check_markdown,
    )

    try:
        check = check_ros_remote_smoke(
            action_log=output_dir / action_log_name,
            status_log=output_dir / status_log_name,
            diagnostics_log=output_dir / diagnostics_log_name,
            expected_model=model,
            expected_remote_url=remote_url,
            require_actions=require_actions,
            require_diagnostics=require_diagnostics,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    payload = json.dumps(check.to_dict(), indent=2)
    markdown = format_ros_remote_smoke_check_markdown(check, title=title)
    if out is not None:
        _write_text(out, payload + "\n")
    if markdown_out is not None:
        _write_text(markdown_out, markdown)
    if json_output:
        typer.echo(payload)
    else:
        typer.echo(markdown)
    if strict and not check.ok:
        raise typer.Exit(1)


@ros_app.command("remote-smoke-report")
def ros_remote_smoke_report(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Remote model name requested by the ROS2 runtime."),
    ] = "openvla",
    remote_url: Annotated[
        str,
        typer.Option("--remote-url", help="Remote vla-zoo server endpoint."),
    ] = "http://gpu-box:8001",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for JSONL logs and report artifacts."),
    ] = Path("results/ros2_remote_smoke"),
    duration_sec: Annotated[
        float,
        typer.Option(
            "--duration-sec",
            help="How long to run remote_smoke_record.launch.py before reports.",
        ),
    ] = 30.0,
    skip_launch: Annotated[
        bool,
        typer.Option(
            "--skip-launch",
            help="Do not run ROS2; generate reports from existing JSONL logs in output-dir.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--append", help="Remove previous logs before running ROS2."),
    ] = True,
    fastdds_udp: Annotated[
        bool,
        typer.Option(
            "--fastdds-udp/--no-fastdds-udp",
            help="Set FASTDDS_BUILTIN_TRANSPORTS=UDPv4 for the ROS2 launch subprocess.",
        ),
    ] = True,
    instruction: Annotated[
        str,
        typer.Option("--instruction", help="Instruction passed to the smoke input node."),
    ] = "pick up the red block",
    task_id: Annotated[
        str,
        typer.Option("--task-id", help="Typed instruction task_id for the smoke run."),
    ] = "ros2_remote_smoke_pick_red_block",
    publish_actions_in_dry_run: Annotated[
        bool,
        typer.Option(
            "--publish-actions-in-dry-run/--suppress-actions-in-dry-run",
            help="Publish typed action messages while dry_run remains true.",
        ),
    ] = True,
    action_log_name: Annotated[
        str,
        typer.Option("--action-log-name", help="Action JSONL filename inside output-dir."),
    ] = "vla_actions.jsonl",
    status_log_name: Annotated[
        str,
        typer.Option("--status-log-name", help="Status JSONL filename inside output-dir."),
    ] = "vla_status.jsonl",
    diagnostics_log_name: Annotated[
        str,
        typer.Option(
            "--diagnostics-log-name",
            help="Diagnostics JSONL filename inside output-dir.",
        ),
    ] = "vla_diagnostics.jsonl",
    dashboard_name: Annotated[
        str,
        typer.Option("--dashboard-name", help="Dashboard HTML filename inside output-dir."),
    ] = "dashboard.html",
    bundle_name: Annotated[
        str,
        typer.Option("--bundle-name", help="Zip report bundle filename inside output-dir."),
    ] = "report_bundle.zip",
    action_trace_name: Annotated[
        str,
        typer.Option("--action-trace-name", help="Action trace HTML filename inside output-dir."),
    ] = "action_trace.html",
    action_analysis_name: Annotated[
        str,
        typer.Option(
            "--action-analysis-name",
            help="Action analysis JSON filename inside output-dir.",
        ),
    ] = "action_analysis.json",
    action_analysis_markdown_name: Annotated[
        str,
        typer.Option(
            "--action-analysis-markdown-name",
            help="Action analysis Markdown filename inside output-dir.",
        ),
    ] = "action_analysis.md",
    remote_check_name: Annotated[
        str,
        typer.Option("--remote-check-name", help="Remote smoke check JSON filename."),
    ] = "remote_smoke_check.json",
    remote_check_markdown_name: Annotated[
        str,
        typer.Option("--remote-check-markdown-name", help="Remote smoke check Markdown filename."),
    ] = "remote_smoke_check.md",
    launch_log_name: Annotated[
        str,
        typer.Option("--launch-log-name", help="ROS2 launch log filename inside output-dir."),
    ] = "launch.log",
    title: Annotated[
        str,
        typer.Option("--title", help="Dashboard/report title."),
    ] = "vla_zoo ROS2 Remote Smoke Report",
) -> None:
    """Run ROS2 remote smoke recording and build dashboard/report artifacts."""

    if duration_sec <= 0 and not skip_launch:
        raise typer.BadParameter("--duration-sec must be positive unless --skip-launch is used.")

    output_dir.mkdir(parents=True, exist_ok=True)
    action_log = output_dir / action_log_name
    status_log = output_dir / status_log_name
    diagnostics_log = output_dir / diagnostics_log_name
    dashboard_out = output_dir / dashboard_name
    bundle_out = output_dir / bundle_name
    action_trace_out = output_dir / action_trace_name
    action_analysis_out = output_dir / action_analysis_name
    action_analysis_markdown_out = output_dir / action_analysis_markdown_name
    remote_check_out = output_dir / remote_check_name
    remote_check_markdown_out = output_dir / remote_check_markdown_name
    launch_log = output_dir / launch_log_name

    if not skip_launch:
        ros2_path = shutil.which("ros2")
        if ros2_path is None:
            typer.echo(
                "ros2 was not found on PATH. Build/source the ROS2 workspace, "
                "or use --skip-launch.",
                err=True,
            )
            raise typer.Exit(1)
        if overwrite:
            for path in (
                action_log,
                status_log,
                diagnostics_log,
                dashboard_out,
                bundle_out,
                action_trace_out,
                action_analysis_out,
                action_analysis_markdown_out,
                remote_check_out,
                remote_check_markdown_out,
                launch_log,
            ):
                _unlink_if_exists(path)

        env = os.environ.copy()
        if fastdds_udp:
            env["FASTDDS_BUILTIN_TRANSPORTS"] = "UDPv4"
        command = [
            ros2_path,
            "launch",
            "vla_zoo",
            "remote_smoke_record.launch.py",
            f"model_name:={model}",
            f"remote_url:={remote_url}",
            f"output_dir:={output_dir}",
            f"action_log_name:={action_log_name}",
            f"status_log_name:={status_log_name}",
            f"diagnostics_log_name:={diagnostics_log_name}",
            f"instruction:={instruction}",
            f"task_id:={task_id}",
            "dry_run:=true",
            f"publish_actions_in_dry_run:={str(publish_actions_in_dry_run).lower()}",
        ]
        typer.echo(f"Running: {' '.join(quote(part) for part in command)}")
        returncode, completed_early = _run_process_for_duration(
            command,
            duration_sec=duration_sec,
            log_path=launch_log,
            env=env,
        )
        if completed_early and returncode != 0:
            typer.echo(
                f"ROS2 launch exited early with code {returncode}. See {launch_log}",
                err=True,
            )
            raise typer.Exit(returncode)

    _finalize_ros_report(
        output_dir=output_dir,
        action_log=action_log,
        status_log=status_log,
        diagnostics_log=diagnostics_log,
        dashboard_out=dashboard_out,
        bundle_out=bundle_out,
        action_trace_out=action_trace_out,
        action_analysis_out=action_analysis_out,
        action_analysis_markdown_out=action_analysis_markdown_out,
        launch_log=launch_log,
        title=title,
        label="ROS2 remote smoke report",
    )
    ros_remote_smoke_check(
        output_dir=output_dir,
        model=model,
        remote_url=remote_url,
        action_log_name=action_log_name,
        status_log_name=status_log_name,
        diagnostics_log_name=diagnostics_log_name,
        out=remote_check_out,
        markdown_out=remote_check_markdown_out,
        require_actions=True,
        require_diagnostics=True,
        json_output=False,
        strict=True,
        title=f"{title}: Remote Runtime Check",
    )


@ros_app.command("smoke-report")
def ros_smoke_report(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for JSONL logs and report artifacts."),
    ] = Path("results/ros2_smoke"),
    duration_sec: Annotated[
        float,
        typer.Option(
            "--duration-sec",
            help="How long to run smoke_record.launch.py before generating reports.",
        ),
    ] = 30.0,
    skip_launch: Annotated[
        bool,
        typer.Option(
            "--skip-launch",
            help="Do not run ROS2; generate reports from existing JSONL logs in output-dir.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--append", help="Remove previous logs before running ROS2."),
    ] = True,
    fastdds_udp: Annotated[
        bool,
        typer.Option(
            "--fastdds-udp/--no-fastdds-udp",
            help="Set FASTDDS_BUILTIN_TRANSPORTS=UDPv4 for the ROS2 launch subprocess.",
        ),
    ] = True,
    instruction: Annotated[
        str,
        typer.Option("--instruction", help="Instruction passed to the smoke input node."),
    ] = "pick up the red block",
    task_id: Annotated[
        str,
        typer.Option("--task-id", help="Typed instruction task_id for the smoke run."),
    ] = "ros2_smoke_pick_red_block",
    action_log_name: Annotated[
        str,
        typer.Option("--action-log-name", help="Action JSONL filename inside output-dir."),
    ] = "vla_actions.jsonl",
    status_log_name: Annotated[
        str,
        typer.Option("--status-log-name", help="Status JSONL filename inside output-dir."),
    ] = "vla_status.jsonl",
    diagnostics_log_name: Annotated[
        str,
        typer.Option(
            "--diagnostics-log-name",
            help="Diagnostics JSONL filename inside output-dir.",
        ),
    ] = "vla_diagnostics.jsonl",
    dashboard_name: Annotated[
        str,
        typer.Option("--dashboard-name", help="Dashboard HTML filename inside output-dir."),
    ] = "dashboard.html",
    bundle_name: Annotated[
        str,
        typer.Option("--bundle-name", help="Zip report bundle filename inside output-dir."),
    ] = "report_bundle.zip",
    action_trace_name: Annotated[
        str,
        typer.Option("--action-trace-name", help="Action trace HTML filename inside output-dir."),
    ] = "action_trace.html",
    action_analysis_name: Annotated[
        str,
        typer.Option(
            "--action-analysis-name",
            help="Action analysis JSON filename inside output-dir.",
        ),
    ] = "action_analysis.json",
    action_analysis_markdown_name: Annotated[
        str,
        typer.Option(
            "--action-analysis-markdown-name",
            help="Action analysis Markdown filename inside output-dir.",
        ),
    ] = "action_analysis.md",
    launch_log_name: Annotated[
        str,
        typer.Option("--launch-log-name", help="ROS2 launch log filename inside output-dir."),
    ] = "launch.log",
    title: Annotated[
        str,
        typer.Option("--title", help="Dashboard/report title."),
    ] = "vla_zoo ROS2 Smoke Report",
) -> None:
    """Run ROS2 smoke recording and build dashboard/report artifacts."""

    if duration_sec <= 0 and not skip_launch:
        raise typer.BadParameter("--duration-sec must be positive unless --skip-launch is used.")

    output_dir.mkdir(parents=True, exist_ok=True)
    action_log = output_dir / action_log_name
    status_log = output_dir / status_log_name
    diagnostics_log = output_dir / diagnostics_log_name
    dashboard_out = output_dir / dashboard_name
    bundle_out = output_dir / bundle_name
    action_trace_out = output_dir / action_trace_name
    action_analysis_out = output_dir / action_analysis_name
    action_analysis_markdown_out = output_dir / action_analysis_markdown_name
    launch_log = output_dir / launch_log_name

    if not skip_launch:
        ros2_path = shutil.which("ros2")
        if ros2_path is None:
            typer.echo(
                "ros2 was not found on PATH. Build/source the ROS2 workspace, "
                "or use --skip-launch.",
                err=True,
            )
            raise typer.Exit(1)
        if overwrite:
            for path in (
                action_log,
                status_log,
                diagnostics_log,
                dashboard_out,
                bundle_out,
                action_trace_out,
                action_analysis_out,
                action_analysis_markdown_out,
                launch_log,
            ):
                _unlink_if_exists(path)

        env = os.environ.copy()
        if fastdds_udp:
            env["FASTDDS_BUILTIN_TRANSPORTS"] = "UDPv4"
        command = [
            ros2_path,
            "launch",
            "vla_zoo",
            "smoke_record.launch.py",
            f"output_dir:={output_dir}",
            f"action_log_name:={action_log_name}",
            f"status_log_name:={status_log_name}",
            f"diagnostics_log_name:={diagnostics_log_name}",
            f"instruction:={instruction}",
            f"task_id:={task_id}",
        ]
        typer.echo(f"Running: {' '.join(quote(part) for part in command)}")
        returncode, completed_early = _run_process_for_duration(
            command,
            duration_sec=duration_sec,
            log_path=launch_log,
            env=env,
        )
        if completed_early and returncode != 0:
            typer.echo(
                f"ROS2 launch exited early with code {returncode}. See {launch_log}",
                err=True,
            )
            raise typer.Exit(returncode)

    _finalize_ros_report(
        output_dir=output_dir,
        action_log=action_log,
        status_log=status_log,
        diagnostics_log=diagnostics_log,
        dashboard_out=dashboard_out,
        bundle_out=bundle_out,
        action_trace_out=action_trace_out,
        action_analysis_out=action_analysis_out,
        action_analysis_markdown_out=action_analysis_markdown_out,
        launch_log=launch_log,
        title=title,
        label="ROS2 smoke report",
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
    ] = "dummy,scripted,random,openvla,pi0,smolvla,groot",
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
        f"{'task':<7} {'lift':<5} {'goal_m':>8} {'moved_m':>8} "
        f"{'mean_ms':>9} {'max_ms':>9} {'mean|a|':>9} note"
    )
    typer.echo(
        f"{'-' * 11} {'-' * 5} {'-' * 6} {'-' * 7} {'-' * 6} "
        f"{'-' * 7} {'-' * 5} {'-' * 8} {'-' * 8} "
        f"{'-' * 9} {'-' * 9} {'-' * 9} {'-' * 32}"
    )
    for result in results:
        typer.echo(
            f"{result.model_name:<11} "
            f"{str(result.ok):<5} "
            f"{result.frames:>6} "
            f"{result.adapter_queries:>7} "
            f"{result.adapter_errors:>6} "
            f"{('success' if result.task_success else 'miss'):<7} "
            f"{str(result.cube_lifted):<5} "
            f"{_format_optional_float(result.final_cube_distance_to_goal):>8} "
            f"{_format_optional_float(result.cube_moved_distance):>8} "
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


@compare_app.command("tasks")
def compare_tasks(
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated adapters to verify across the PyBullet task suite.",
        ),
    ] = "dummy,scripted,random,openvla,pi0,smolvla,groot",
    tasks: Annotated[
        str,
        typer.Option(
            "--tasks",
            help="Comma-separated task ids, or 'all' for the built-in PyBullet task suite.",
        ),
    ] = "all",
    runtime: Annotated[str, typer.Option("--runtime")] = "local",
    remote_url: Annotated[str, typer.Option("--remote-url")] = "http://localhost:8000",
    remote_map: Annotated[
        str | None,
        typer.Option(
            "--remote-map",
            help="Comma-separated model=url overrides for remote comparisons.",
        ),
    ] = None,
    model_call_every: Annotated[int, typer.Option("--model-call-every")] = 12,
    render_stride: Annotated[int, typer.Option("--render-stride")] = 24,
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
        typer.Option("--markdown-out", help="Write a Markdown verification report."),
    ] = None,
    html_out: Annotated[
        Path | None,
        typer.Option("--html-out", help="Write a self-contained HTML verification report."),
    ] = None,
) -> None:
    """Run several PyBullet runtime verification tasks per adapter."""

    from vla_zoo.demo.pybullet import (
        PyBulletComparisonTarget,
        compare_pybullet_task_suite,
        default_pybullet_tasks,
        format_pybullet_comparison_html,
        format_pybullet_comparison_markdown,
        pybullet_task_by_id,
    )

    model_names = [item.strip() for item in models.split(",") if item.strip()]
    if not model_names:
        raise typer.BadParameter("At least one model name is required.")
    remote_urls = _parse_remote_map(remote_map)
    targets = [
        PyBulletComparisonTarget(
            model_name=model_name,
            runtime=runtime,
            remote_url=remote_urls.get(model_name.strip().lower(), remote_url),
        )
        for model_name in model_names
    ]
    if tasks.strip().lower() == "all":
        task_specs = default_pybullet_tasks()
    else:
        task_specs = [pybullet_task_by_id(item) for item in tasks.split(",") if item.strip()]
    if not task_specs:
        raise typer.BadParameter("At least one task id is required.")

    results = compare_pybullet_task_suite(
        targets,
        task_specs,
        model_call_every=model_call_every,
        render_stride=render_stride,
        allow_local_heavy=allow_local_heavy,
    )
    title = "PyBullet Multi-Task VLA Runtime Verification"
    json_payload = json.dumps([asdict(result) for result in results], indent=2)
    if out is not None:
        _write_text(out, f"{json_payload}\n")
    if markdown_out is not None:
        _write_text(
            markdown_out,
            format_pybullet_comparison_markdown(results, title=title),
        )
    if html_out is not None:
        _write_text(
            html_out,
            format_pybullet_comparison_html(results, title=title),
        )
    if json_output:
        typer.echo(json_payload)
        return

    typer.echo(
        f"{'task':<21} {'model':<11} {'ok':<5} {'queries':>7} {'errors':>6} "
        f"{'scene':<7} {'goal_m':>8} {'mean_ms':>9} {'mean|a|':>9} note"
    )
    typer.echo(
        f"{'-' * 21} {'-' * 11} {'-' * 5} {'-' * 7} {'-' * 6} "
        f"{'-' * 7} {'-' * 8} {'-' * 9} {'-' * 9} {'-' * 32}"
    )
    for result in results:
        typer.echo(
            f"{result.task_id:<21} "
            f"{result.model_name:<11} "
            f"{str(result.ok):<5} "
            f"{result.adapter_queries:>7} "
            f"{result.adapter_errors:>6} "
            f"{('success' if result.task_success else 'miss'):<7} "
            f"{_format_optional_float(result.final_cube_distance_to_goal):>8} "
            f"{_format_optional_float(result.mean_latency_ms):>9} "
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
    instruction: Annotated[
        str | None,
        typer.Option("--instruction", "-i", help="Override the task instruction."),
    ] = None,
    task: Annotated[
        str,
        typer.Option("--task", help="PyBullet task id, for example pick_red_block."),
    ] = "pick_red_block",
    out: Annotated[Path, typer.Option("--out", "-o")] = Path(
        "docs/assets/simulation_pick_place.gif"
    ),
    model_call_every: Annotated[int, typer.Option("--model-call-every")] = 8,
    render_stride: Annotated[
        int,
        typer.Option("--render-stride", help="Render every N scripted simulation steps."),
    ] = 3,
) -> None:
    """Render the PyBullet pick-and-place demo with any VLA adapter."""

    from vla_zoo.demo.pybullet import PyBulletDemoConfig, pybullet_task_by_id, render_pybullet_demo

    try:
        task_spec = pybullet_task_by_id(task)
        config_kwargs: dict[str, object] = {
            "model_name": model,
            "runtime": runtime,
            "remote_url": remote_url,
            "out": out,
            "model_call_every": model_call_every,
            "render_stride": render_stride,
        }
        if instruction:
            config_kwargs["instruction"] = instruction
        result = render_pybullet_demo(
            PyBulletDemoConfig.from_task(
                task_spec,
                **config_kwargs,
            )
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    typer.echo(json.dumps(result, indent=2))


def _coerce_kwarg_value(raw: str) -> object:
    """Coerce a CLI string into bool/int/float, falling back to the raw string."""

    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_adapter_kwargs(pairs: list[str] | None) -> dict[str, object]:
    """Parse repeated ``key=value`` options into a typed adapter-kwargs dict."""

    result: dict[str, object] = {}
    for pair in pairs or []:
        if "=" not in pair:
            msg = f"adapter kwarg must be key=value, got {pair!r}"
            raise ValueError(msg)
        key, _, value = pair.partition("=")
        result[key.strip()] = _coerce_kwarg_value(value)
    return result


@demo_app.command("action-probe")
def demo_action_probe(
    model: Annotated[str, typer.Option("--model", "-m")] = "dummy",
    runtime: Annotated[str, typer.Option("--runtime")] = "local",
    remote_url: Annotated[str, typer.Option("--remote-url")] = "http://localhost:8000",
    instruction: Annotated[
        str | None,
        typer.Option("--instruction", "-i", help="Override the task instruction."),
    ] = None,
    task: Annotated[
        str,
        typer.Option("--task", help="PyBullet task id, for example pick_red_block."),
    ] = "pick_red_block",
    out: Annotated[Path, typer.Option("--out", "-o")] = Path(
        "results/pybullet_action_probe.jsonl"
    ),
    summary_md: Annotated[
        Path | None,
        typer.Option("--summary-md", help="Write the runtime-evidence summary as Markdown."),
    ] = None,
    summary_json: Annotated[
        Path | None,
        typer.Option("--summary-json", help="Write the runtime-evidence summary as JSON."),
    ] = None,
    model_call_every: Annotated[int, typer.Option("--model-call-every")] = 4,
    render_stride: Annotated[
        int,
        typer.Option("--render-stride", help="Render every N scripted simulation steps."),
    ] = 6,
    allow_local_heavy: Annotated[
        bool,
        typer.Option(
            "--allow-local-heavy",
            help="Permit local heavy adapters (openvla/smolvla); they download/load weights.",
        ),
    ] = False,
    pretrained: Annotated[
        str | None,
        typer.Option("--pretrained", help="Adapter pretrained checkpoint id or path."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Adapter device override, for example cuda or cpu."),
    ] = None,
    local_files_only: Annotated[
        bool,
        typer.Option("--local-files-only", help="Load adapter weights from local cache only."),
    ] = False,
    return_action_chunk: Annotated[
        bool,
        typer.Option(
            "--return-action-chunk",
            help="Ask the adapter for a full action chunk per query (fresh encode each call).",
        ),
    ] = False,
    adapter_kwarg: Annotated[
        list[str] | None,
        typer.Option(
            "--adapter-kwarg",
            help="Extra adapter kwarg as key=value (repeatable); coerced to bool/int/float/str.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the summary as JSON."),
    ] = False,
) -> None:
    """Record an adapter's action stream on real PyBullet-rendered scene frames.

    A runtime-path probe: it drives the adapter on genuinely rendered frames (exercising the
    real image preprocessing path) and records latency + action magnitude as a canonical
    ``vla_actions.jsonl`` log. It makes no task-success or policy-quality claim.
    """

    from vla_zoo.demo.action_probe import (
        format_action_probe_summary_markdown,
        record_pybullet_action_probe,
        write_action_probe_log,
    )
    from vla_zoo.demo.pybullet import (
        HEAVY_LOCAL_MODELS,
        PyBulletDemoConfig,
        pybullet_task_by_id,
    )

    canonical = model.strip().lower()
    if runtime == "local" and canonical in HEAVY_LOCAL_MODELS and not allow_local_heavy:
        typer.echo(
            f"local heavy adapter {model!r} skipped to avoid model download; "
            "use --allow-local-heavy or --runtime remote",
            err=True,
        )
        raise typer.Exit(1)

    try:
        adapter_kwargs = _parse_adapter_kwargs(adapter_kwarg)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if pretrained:
        adapter_kwargs["pretrained"] = pretrained
    if device:
        adapter_kwargs["device"] = device
    if local_files_only:
        adapter_kwargs["local_files_only"] = True
    if return_action_chunk:
        adapter_kwargs["return_action_chunk"] = True

    try:
        task_spec = pybullet_task_by_id(task)
        config_kwargs: dict[str, object] = {
            "model_name": model,
            "runtime": runtime,
            "remote_url": remote_url,
            "out": out,
            "model_call_every": model_call_every,
            "render_stride": render_stride,
        }
        if instruction:
            config_kwargs["instruction"] = instruction
        if adapter_kwargs:
            config_kwargs["adapter_kwargs"] = adapter_kwargs
        summary, records = record_pybullet_action_probe(
            PyBulletDemoConfig.from_task(task_spec, **config_kwargs)
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    write_action_probe_log(out, records)
    typer.echo(f"Action log written to {out} ({len(records)} records)")
    if summary_json is not None:
        _write_text(summary_json, json.dumps(summary.to_dict(), indent=2) + "\n")
        typer.echo(f"Summary JSON written to {summary_json}")
    if summary_md is not None:
        _write_text(summary_md, format_action_probe_summary_markdown(summary))
        typer.echo(f"Summary Markdown written to {summary_md}")

    if not json_output:
        p50 = "n/a" if summary.latency_ms_p50 is None else f"{summary.latency_ms_p50:.1f} ms"
        typer.echo(
            f"{summary.model}: {summary.record_count} queries, "
            f"latency p50 {p50} (policy_quality={summary.policy_quality})"
        )
    typer.echo(json.dumps(summary.to_dict(), indent=2))


@demo_app.command("gif-suite")
def demo_gif_suite(
    models: Annotated[
        str,
        typer.Option("--models", "-m", help="Comma-separated models to render."),
    ] = "dummy,scripted,random",
    tasks: Annotated[
        str,
        typer.Option("--tasks", help="Comma-separated PyBullet task ids, or 'all'."),
    ] = "all",
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", "-o", help="Directory for generated GIFs and reports."),
    ] = Path("docs/assets/gif_suite"),
    runtime: Annotated[str, typer.Option("--runtime")] = "local",
    remote_url: Annotated[str, typer.Option("--remote-url")] = "http://localhost:8000",
    model_call_every: Annotated[int, typer.Option("--model-call-every")] = 8,
    render_stride: Annotated[int, typer.Option("--render-stride")] = 8,
    manifest_out: Annotated[
        Path | None,
        typer.Option("--manifest-out", help="Manifest JSON path. Defaults under out-dir."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Markdown gallery path. Defaults under out-dir."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Render a README-ready suite of real PyBullet simulation GIFs."""

    from vla_zoo.demo.gif_suite import (
        build_pybullet_gif_specs,
        render_pybullet_gif_suite,
        resolve_pybullet_tasks,
        write_gif_gallery,
        write_gif_manifest,
    )

    model_names = _parse_name_list(models)
    task_specs = resolve_pybullet_tasks(tasks)
    specs = build_pybullet_gif_specs(
        models=model_names,
        tasks=task_specs,
        out_dir=out_dir,
        runtime=runtime,
        remote_url=remote_url,
        model_call_every=model_call_every,
        render_stride=render_stride,
    )
    results = render_pybullet_gif_suite(specs)
    manifest_path = manifest_out if manifest_out is not None else out_dir / "gif_manifest.json"
    markdown_path = markdown_out if markdown_out is not None else out_dir / "README.md"
    write_gif_manifest(manifest_path, results)
    write_gif_gallery(markdown_path, results)

    payload = {
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "markdown": str(markdown_path),
        "results": [result.to_dict() for result in results],
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        ok_count = sum(1 for result in results if result.ok)
        typer.echo(f"GIF suite written to {out_dir} ({ok_count}/{len(results)} ok)")
        typer.echo(f"Manifest: {manifest_path}")
        typer.echo(f"Markdown: {markdown_path}")


@demo_app.command("gif-check")
def demo_gif_check(
    path: Annotated[
        Path,
        typer.Argument(help="GIF suite directory or gif_manifest.json path."),
    ] = Path("docs/assets/gif_suite"),
    expected_width: Annotated[int, typer.Option("--expected-width")] = 960,
    expected_height: Annotated[int, typer.Option("--expected-height")] = 540,
    min_frames: Annotated[int, typer.Option("--min-frames")] = 12,
    min_bytes: Annotated[int, typer.Option("--min-bytes")] = 1024,
    link_files: Annotated[
        str | None,
        typer.Option(
            "--link-files",
            help="Comma-separated README/Page files whose local GIF/gallery links should exist.",
        ),
    ] = "README.md,docs/index.html,docs/assets/gif_suite/README.md",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write JSON check report."),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Write Markdown check report."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option("--strict/--no-strict", help="Exit non-zero when GIF checks fail."),
    ] = True,
) -> None:
    """Validate PyBullet GIF assets, manifest consistency, and README/Page links."""

    from vla_zoo.demo.gif_suite import check_gif_suite, format_gif_check_markdown

    report = check_gif_suite(
        path,
        expected_width=expected_width,
        expected_height=expected_height,
        min_frames=min_frames,
        min_bytes=min_bytes,
        link_files=_parse_optional_paths(link_files),
    )
    payload = json.dumps(report.to_dict(), indent=2)
    markdown = format_gif_check_markdown(report)
    if out is not None:
        _write_text(out, f"{payload}\n")
    if markdown_out is not None:
        _write_text(markdown_out, markdown)
    if json_output:
        typer.echo(payload)
    else:
        typer.echo(
            f"{'gif':<48} {'status':<8} {'frames':<6} {'resolution':<11} size"
        )
        typer.echo(f"{'-' * 48} {'-' * 8} {'-' * 6} {'-' * 11} {'-' * 10}")
        for asset in report.assets:
            typer.echo(
                f"{Path(asset.path).name:<48} "
                f"{('ok' if asset.ok else 'failed'):<8} "
                f"{asset.frames:<6} "
                f"{asset.width}x{asset.height:<7} "
                f"{asset.bytes}"
            )
        if report.issues:
            typer.echo("Suite issues:")
            for issue in report.issues:
                typer.echo(f"- {issue.level}: {issue.message}")
    if out is not None:
        typer.echo(f"JSON written to {out}")
    if markdown_out is not None:
        typer.echo(f"Markdown written to {markdown_out}")
    if strict and not report.ok:
        raise typer.Exit(1)


@demo_app.command("gif-report")
def demo_gif_report(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="GIF suite manifest JSON path."),
    ] = Path("docs/assets/gif_suite/gif_manifest.json"),
    expected_width: Annotated[int, typer.Option("--expected-width")] = 960,
    expected_height: Annotated[int, typer.Option("--expected-height")] = 540,
    min_frames: Annotated[int, typer.Option("--min-frames")] = 12,
    min_bytes: Annotated[int, typer.Option("--min-bytes")] = 1024,
    html_out: Annotated[
        Path,
        typer.Option("--html-out", help="Output HTML gallery path."),
    ] = Path("docs/assets/gif_suite/index.html"),
    check_json_out: Annotated[
        Path | None,
        typer.Option("--check-json-out", help="Optional JSON check report path."),
    ] = None,
    link_files: Annotated[
        str | None,
        typer.Option("--link-files", help="Comma-separated README/Page link files to check."),
    ] = "README.md,docs/index.html,docs/assets/gif_suite/README.md",
    title: Annotated[
        str,
        typer.Option("--title", help="HTML report title."),
    ] = "vla_zoo PyBullet GIF Gallery",
    strict: Annotated[
        bool,
        typer.Option("--strict/--no-strict", help="Exit non-zero when GIF checks fail."),
    ] = True,
) -> None:
    """Build a static HTML gallery and QA report for generated PyBullet GIFs."""

    from vla_zoo.demo.gif_suite import check_gif_suite, write_gif_report_html

    report = check_gif_suite(
        manifest,
        expected_width=expected_width,
        expected_height=expected_height,
        min_frames=min_frames,
        min_bytes=min_bytes,
        link_files=_parse_optional_paths(link_files),
    )
    write_gif_report_html(html_out, report, title=title)
    if check_json_out is not None:
        _write_text(check_json_out, json.dumps(report.to_dict(), indent=2) + "\n")
    typer.echo(f"HTML written to {html_out}")
    if check_json_out is not None:
        typer.echo(f"JSON written to {check_json_out}")
    if strict and not report.ok:
        raise typer.Exit(1)


@demo_app.command("action-playground")
def demo_action_playground(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="GIF suite manifest JSON path."),
    ] = Path("docs/assets/gif_suite/gif_manifest.json"),
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output static HTML playground path."),
    ] = Path("docs/assets/action_playground.html"),
    trace_out: Annotated[
        Path,
        typer.Option("--trace-out", help="Output machine-readable action trace JSON path."),
    ] = Path("docs/assets/action_playground.json"),
    title: Annotated[
        str,
        typer.Option("--title", help="HTML report title."),
    ] = "vla_zoo Action Playground",
    max_records: Annotated[
        int | None,
        typer.Option("--max-records", help="Limit records while developing reports."),
    ] = None,
    allow_local_heavy: Annotated[
        bool,
        typer.Option(
            "--allow-local-heavy",
            help="Allow local heavy adapters such as OpenVLA to load real model weights.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable summary."),
    ] = False,
) -> None:
    """Build a static playground that scrubs PyBullet action traces beside GIFs."""

    from vla_zoo.demo.action_playground import (
        build_action_playground_records,
        write_action_playground_html,
        write_action_playground_trace,
    )

    try:
        records = build_action_playground_records(
            manifest,
            max_records=max_records,
            allow_local_heavy=allow_local_heavy,
        )
        write_action_playground_trace(trace_out, records)
        write_action_playground_html(out, records, title=title)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    ok_count = sum(1 for record in records if record.ok)
    payload = {
        "manifest": str(manifest),
        "html": str(out),
        "trace": str(trace_out),
        "records": len(records),
        "ok": ok_count,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(f"Action playground written to {out} ({ok_count}/{len(records)} ok)")
        typer.echo(f"Trace JSON written to {trace_out}")


@demo_app.command("action-playground-record")
def demo_action_playground_record(
    models: Annotated[
        str,
        typer.Option(
            "--model",
            "--models",
            "-m",
            help="Comma-separated adapters to record, for example smolvla or openvla,pi0.",
        ),
    ] = "dummy",
    tasks: Annotated[
        str,
        typer.Option("--tasks", help="Comma-separated PyBullet task ids, or 'all'."),
    ] = "all",
    runtime: Annotated[str, typer.Option("--runtime")] = "remote",
    remote_url: Annotated[str, typer.Option("--remote-url")] = "http://localhost:8000",
    remote_map: Annotated[
        str | None,
        typer.Option(
            "--remote-map",
            help="Comma-separated model=url overrides, for example openvla=http://gpu:8001.",
        ),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output action_playground.json trace path."),
    ] = Path("results/action_playground_record.json"),
    html_out: Annotated[
        Path | None,
        typer.Option("--html-out", help="Optional Action Playground HTML output path."),
    ] = None,
    reference_gif_dir: Annotated[
        Path,
        typer.Option("--reference-gif-dir", help="Directory containing reference PyBullet GIFs."),
    ] = Path("docs/assets/gif_suite"),
    reference_gif_model: Annotated[
        str,
        typer.Option(
            "--reference-gif-model",
            help="Existing GIF model suffix used as the scene reference.",
        ),
    ] = "scripted",
    model_call_every: Annotated[int, typer.Option("--model-call-every")] = 8,
    render_stride: Annotated[int, typer.Option("--render-stride")] = 8,
    max_records: Annotated[
        int | None,
        typer.Option("--max-records", help="Limit model/task records while developing."),
    ] = None,
    allow_local_heavy: Annotated[
        bool,
        typer.Option(
            "--allow-local-heavy",
            help="Allow local heavy adapters such as OpenVLA to load real model weights.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable summary."),
    ] = False,
) -> None:
    """Record PyBullet action traces for local or remote VLA adapters without new GIFs."""

    from vla_zoo.demo.action_playground import (
        build_action_playground_task_records,
        write_action_playground_html,
        write_action_playground_trace,
    )
    from vla_zoo.demo.gif_suite import resolve_pybullet_tasks

    try:
        records = build_action_playground_task_records(
            models=_parse_name_list(models),
            tasks=resolve_pybullet_tasks(tasks),
            out_dir=reference_gif_dir,
            runtime=runtime,
            remote_url=remote_url,
            remote_urls=_parse_remote_map(remote_map),
            model_call_every=model_call_every,
            render_stride=render_stride,
            reference_gif_model=reference_gif_model,
            max_records=max_records,
            allow_local_heavy=allow_local_heavy,
        )
        write_action_playground_trace(out, records)
        if html_out is not None:
            write_action_playground_html(html_out, records)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    ok_count = sum(1 for record in records if record.ok)
    payload = {
        "trace": str(out),
        "html": str(html_out) if html_out is not None else None,
        "records": len(records),
        "ok": ok_count,
        "runtime": runtime,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(f"Action trace written to {out} ({ok_count}/{len(records)} ok)")
        if html_out is not None:
            typer.echo(f"Action playground HTML written to {html_out}")


@demo_app.command("action-playground-view")
def demo_action_playground_view(
    traces: Annotated[
        str,
        typer.Option(
            "--trace",
            "-t",
            help="Comma-separated action_playground.json files to merge and render.",
        ),
    ] = "docs/assets/action_playground.json",
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output static HTML playground path."),
    ] = Path("docs/assets/action_playground.html"),
    merged_out: Annotated[
        Path | None,
        typer.Option("--merged-out", help="Optional merged trace JSON output path."),
    ] = None,
    title: Annotated[
        str,
        typer.Option("--title", help="HTML report title."),
    ] = "vla_zoo Action Playground",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable summary."),
    ] = False,
) -> None:
    """Render an Action Playground from one or more saved trace JSON files."""

    from vla_zoo.demo.action_playground import (
        load_action_playground_traces,
        write_action_playground_html,
        write_action_playground_trace,
    )

    trace_paths = _parse_paths(traces)
    try:
        records = load_action_playground_traces(trace_paths)
        if merged_out is not None:
            write_action_playground_trace(merged_out, records)
        write_action_playground_html(out, records, title=title)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    ok_count = sum(1 for record in records if record.ok)
    payload = {
        "traces": [str(path) for path in trace_paths],
        "html": str(out),
        "merged_trace": str(merged_out) if merged_out is not None else None,
        "records": len(records),
        "ok": ok_count,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(f"Action playground HTML written to {out} ({ok_count}/{len(records)} ok)")
        if merged_out is not None:
            typer.echo(f"Merged trace JSON written to {merged_out}")


@demo_app.command("action-playground-check")
def demo_action_playground_check(
    traces: Annotated[
        str,
        typer.Option(
            "--trace",
            "-t",
            help="Comma-separated action_playground.json files to validate.",
        ),
    ] = "docs/assets/action_playground.json",
    out: Annotated[
        Path,
        typer.Option("--out", help="Write machine-readable check JSON."),
    ] = Path("docs/assets/action_playground_check.json"),
    markdown_out: Annotated[
        Path,
        typer.Option("--markdown-out", help="Write README/Page-ready Markdown report."),
    ] = Path("docs/reports/model_comparison.md"),
    expected_models: Annotated[
        str | None,
        typer.Option(
            "--expected-models",
            help="Comma-separated model names expected in the trace. Empty disables this check.",
        ),
    ] = "dummy,scripted,random",
    expected_tasks: Annotated[
        str | None,
        typer.Option(
            "--expected-tasks",
            help="Comma-separated PyBullet task ids, or 'all'. Empty disables this check.",
        ),
    ] = "all",
    base_dir: Annotated[
        Path,
        typer.Option("--base-dir", help="Base directory for resolving relative GIF paths."),
    ] = Path("."),
    min_frames: Annotated[
        int,
        typer.Option("--min-frames", help="Minimum frames required per trace record."),
    ] = 12,
    require_gifs: Annotated[
        bool,
        typer.Option("--require-gifs/--no-require-gifs", help="Require referenced GIF files."),
    ] = True,
    strict: Annotated[
        bool,
        typer.Option("--strict/--no-strict", help="Exit non-zero when validation fails."),
    ] = True,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable report."),
    ] = False,
) -> None:
    """Validate Action Playground traces and write a model/task comparison report."""

    from vla_zoo.demo.action_playground import (
        check_action_playground_traces,
        format_action_playground_check_markdown,
    )
    from vla_zoo.demo.gif_suite import resolve_pybullet_tasks

    trace_paths = _parse_paths(traces)
    expected_task_ids = (
        [task.task_id for task in resolve_pybullet_tasks(expected_tasks)]
        if expected_tasks
        else []
    )
    try:
        report = check_action_playground_traces(
            trace_paths,
            base_dir=base_dir,
            expected_models=_parse_optional_name_list(expected_models),
            expected_tasks=expected_task_ids,
            min_frames=min_frames,
            require_gifs=require_gifs,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    payload = json.dumps(report.to_dict(), indent=2)
    markdown = format_action_playground_check_markdown(report)
    _write_text(out, payload + "\n")
    _write_text(markdown_out, markdown)

    if json_output:
        typer.echo(payload)
    else:
        typer.echo(
            f"Action playground check: {report.ok_count}/{report.record_count} records ok"
        )
        typer.echo(f"JSON written to {out}")
        typer.echo(f"Markdown written to {markdown_out}")
        if report.issues:
            typer.echo("Issues:")
            for issue in report.issues:
                typer.echo(f"- {issue}")
    if strict and not report.ok:
        raise typer.Exit(1)


@demo_app.command("action-playground-remote-smoke")
def demo_action_playground_remote_smoke(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Single adapter hosted by the temporary server."),
    ] = "dummy",
    tasks: Annotated[
        str,
        typer.Option("--tasks", help="Comma-separated PyBullet task ids, or 'all'."),
    ] = "all",
    host: Annotated[
        str,
        typer.Option("--host", help="Temporary server bind host."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="Temporary server port. Use 0 to pick a free port."),
    ] = 0,
    out: Annotated[
        Path,
        typer.Option("--out", help="Remote-only action trace JSON."),
    ] = Path("docs/assets/action_playground_remote_dummy.json"),
    check_out: Annotated[
        Path,
        typer.Option("--check-out", help="Remote-only validation JSON."),
    ] = Path("docs/assets/action_playground_remote_dummy_check.json"),
    markdown_out: Annotated[
        Path,
        typer.Option("--markdown-out", help="Remote runtime smoke Markdown report."),
    ] = Path("docs/reports/remote_runtime_smoke.md"),
    base_trace: Annotated[
        Path | None,
        typer.Option("--base-trace", help="Optional existing local trace to merge with remote."),
    ] = Path("docs/assets/action_playground.json"),
    merged_out: Annotated[
        Path | None,
        typer.Option("--merged-out", help="Optional merged local+remote trace JSON."),
    ] = Path("docs/assets/action_playground_with_remote.json"),
    html_out: Annotated[
        Path | None,
        typer.Option("--html-out", help="Optional merged Action Playground HTML."),
    ] = Path("docs/assets/action_playground_with_remote.html"),
    reference_gif_dir: Annotated[
        Path,
        typer.Option("--reference-gif-dir", help="Directory containing reference PyBullet GIFs."),
    ] = Path("docs/assets/gif_suite"),
    reference_gif_model: Annotated[
        str,
        typer.Option("--reference-gif-model", help="Reference GIF model suffix."),
    ] = "scripted",
    model_call_every: Annotated[int, typer.Option("--model-call-every")] = 8,
    render_stride: Annotated[int, typer.Option("--render-stride")] = 8,
    min_frames: Annotated[int, typer.Option("--min-frames")] = 12,
    startup_timeout_sec: Annotated[
        float,
        typer.Option("--startup-timeout-sec", help="Seconds to wait for /health."),
    ] = 20.0,
    log_path: Annotated[
        Path,
        typer.Option("--log-path", help="Temporary server stdout/stderr log."),
    ] = Path("results/action_playground_remote_smoke/server.log"),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable summary."),
    ] = False,
) -> None:
    """Start a temporary dummy server and record Action Playground traces over HTTP."""

    from vla_zoo.demo.action_playground import (
        build_action_playground_task_records,
        check_action_playground_records,
        format_action_playground_check_markdown,
        load_action_playground_trace,
        merge_action_playground_records,
        write_action_playground_html,
        write_action_playground_trace,
    )
    from vla_zoo.demo.gif_suite import resolve_pybullet_tasks

    selected_port = _free_tcp_port(host) if port == 0 else port
    remote_url = f"http://{_client_host_for_bind_host(host)}:{selected_port}"
    command = [
        sys.executable,
        "-m",
        "vla_zoo.cli.main",
        "serve",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(selected_port),
    ]
    task_specs = resolve_pybullet_tasks(tasks)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        try:
            startup_error = _wait_for_http_health(remote_url, timeout_sec=startup_timeout_sec)
            if startup_error is not None:
                raise RuntimeError(
                    f"Temporary vla-zoo server did not become ready at {remote_url}: "
                    f"{startup_error}. See {log_path}."
                )
            records = build_action_playground_task_records(
                models=(model,),
                tasks=task_specs,
                out_dir=reference_gif_dir,
                runtime="remote",
                remote_url=remote_url,
                model_call_every=model_call_every,
                render_stride=render_stride,
                reference_gif_model=reference_gif_model,
            )
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    write_action_playground_trace(out, records)
    report = check_action_playground_records(
        records,
        trace_paths=(out,),
        expected_models=(model,),
        expected_tasks=tuple(task.task_id for task in task_specs),
        min_frames=min_frames,
        require_gifs=True,
    )
    _write_text(check_out, json.dumps(report.to_dict(), indent=2) + "\n")
    _write_text(
        markdown_out,
        format_action_playground_check_markdown(
            report,
            title="Remote Runtime Smoke Report",
        ),
    )

    merged_records = records
    if base_trace is not None and base_trace.exists():
        merged_records = merge_action_playground_records(
            [*load_action_playground_trace(base_trace), *records]
        )
    if merged_out is not None:
        write_action_playground_trace(merged_out, merged_records)
    if html_out is not None:
        write_action_playground_html(
            html_out,
            merged_records,
            title="vla_zoo Action Playground - Local and Remote Runtime",
        )

    payload = {
        "model": model,
        "runtime": "remote",
        "remote_url": remote_url,
        "trace": str(out),
        "check": str(check_out),
        "markdown": str(markdown_out),
        "merged_trace": str(merged_out) if merged_out is not None else None,
        "html": str(html_out) if html_out is not None else None,
        "records": len(records),
        "ok": report.ok_count,
        "server_log": str(log_path),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(f"Remote Action Playground smoke: {report.ok_count}/{len(records)} ok")
        typer.echo(f"Trace written to {out}")
        typer.echo(f"Markdown written to {markdown_out}")
        if html_out is not None:
            typer.echo(f"HTML written to {html_out}")
        typer.echo(f"Server log written to {log_path}")
    if not report.ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
