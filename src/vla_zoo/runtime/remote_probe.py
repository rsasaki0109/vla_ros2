"""Health-first remote VLA probe.

Checks a remote server's ``/health`` endpoint before sending a single
``/v1/predict`` request, and records the response as a structured artifact. This
is a runtime-path probe: it does not download model weights and makes no
task-success claim. Heavyweight models (OpenVLA 7B) are expected to run on a
remote GPU box; this probe runs on the robot/client side.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass

STATUS_OK = "ok"
STATUS_UNREACHABLE = "unreachable"
STATUS_PREDICT_FAILED = "predict_failed"

HealthFn = Callable[[str, float], tuple[str | None, dict[str, object] | None]]
PredictFn = Callable[[str, str, str, float], dict[str, object]]


@dataclass(frozen=True)
class RemoteProbeResult:
    model_name: str
    remote_url: str
    instruction: str
    status: str
    health: dict[str, object] | None
    action: dict[str, object] | None
    error: str | None

    @property
    def is_ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _default_health(remote_url: str, timeout: float) -> tuple[str | None, dict[str, object] | None]:
    url = f"{remote_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.getcode() != 200:
                return f"health endpoint returned HTTP {response.getcode()}", None
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return str(exc), None
    if not isinstance(payload, dict):
        return "health endpoint returned a non-object payload", None
    if not payload.get("ready", False):
        return "health endpoint reported the server is not ready", payload
    return None, payload


def _default_predict(
    model_name: str,
    remote_url: str,
    instruction: str,
    timeout: float,
) -> dict[str, object]:
    import numpy as np

    from vla_zoo.core.registry import load_model
    from vla_zoo.runtime.schemas import prediction_to_response

    model = load_model(model_name, runtime="remote", remote_url=remote_url, timeout=timeout)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.zeros(7, dtype=np.float32)
    action = model.predict(image=image, instruction=instruction, state=state)
    return prediction_to_response(action).model_dump(mode="json")


def probe_remote_model(
    *,
    model_name: str,
    remote_url: str,
    instruction: str = "pick up the red block",
    timeout: float = 30.0,
    health_fn: HealthFn | None = None,
    predict_fn: PredictFn | None = None,
) -> RemoteProbeResult:
    """Probe a remote server: check /health, then record one /v1/predict response."""

    health_fn = health_fn or _default_health
    predict_fn = predict_fn or _default_predict

    health_error, health_payload = health_fn(remote_url, timeout)
    if health_error is not None:
        return RemoteProbeResult(
            model_name=model_name,
            remote_url=remote_url,
            instruction=instruction,
            status=STATUS_UNREACHABLE,
            health=health_payload,
            action=None,
            error=health_error,
        )

    try:
        action = predict_fn(model_name, remote_url, instruction, timeout)
    except Exception as exc:  # noqa: BLE001 - record any client/runtime failure as a result
        return RemoteProbeResult(
            model_name=model_name,
            remote_url=remote_url,
            instruction=instruction,
            status=STATUS_PREDICT_FAILED,
            health=health_payload,
            action=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    return RemoteProbeResult(
        model_name=model_name,
        remote_url=remote_url,
        instruction=instruction,
        status=STATUS_OK,
        health=health_payload,
        action=action,
        error=None,
    )


def format_remote_probe_markdown(result: RemoteProbeResult) -> str:
    lines = [
        f"# Remote VLA Probe: {result.model_name}",
        "",
        "Health-first remote runtime probe. This records a single `/v1/predict`",
        "response over HTTP. It is not a robot task-success benchmark.",
        "",
        f"- model: `{result.model_name}`",
        f"- remote_url: `{result.remote_url}`",
        f"- instruction: `{result.instruction}`",
        f"- status: `{result.status}`",
    ]
    if result.error:
        lines.append(f"- error: `{result.error}`")
    lines.append("")
    if result.health is not None:
        lines.extend(["## Health", "", "```json", json.dumps(result.health, indent=2), "```", ""])
    if result.action is not None:
        lines.extend(
            ["## Recorded Action", "", "```json", json.dumps(result.action, indent=2), "```", ""]
        )
    return "\n".join(lines)
