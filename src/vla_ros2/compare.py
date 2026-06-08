"""Compare multiple vla_ros2 adapters on the same observation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

from vla_ros2.core.errors import UnknownModelError, VLARos2Error
from vla_ros2.core.registry import get_adapter_info, load_model
from vla_ros2.core.types import VLAAction, VLAActionChunk, VLAObservation

# Per-adapter load kwargs applied unless overridden by ``load_kwargs``.
DEFAULT_LOAD_KWARGS: dict[str, dict[str, Any]] = {
    "pi0": {"enable_local": True},
}


@dataclass(frozen=True)
class ModelCompareResult:
    name: str
    ok: bool
    error: str | None = None
    load_ms: float | None = None
    infer_ms: float | None = None
    action_space: str = ""
    action_shape: tuple[int, ...] = ()
    action_names: tuple[str, ...] = ()
    action_values: tuple[float, ...] = ()
    compare_role: str = ""
    default_checkpoint: str = ""
    experimental: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def action_preview(self) -> str:
        if not self.ok or not self.action_values:
            return ""
        parts = [
            f"{name}={value:.4f}" if name else f"{value:.4f}"
            for name, value in zip(self.action_names, self.action_values, strict=False)
        ]
        if len(parts) > 6:
            return ", ".join(parts[:6]) + ", ..."
        return ", ".join(parts)

    def to_row(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "ok": self.ok,
            "load_ms": round(self.load_ms, 1) if self.load_ms is not None else None,
            "infer_ms": round(self.infer_ms, 1) if self.infer_ms is not None else None,
            "action_shape": list(self.action_shape),
            "action_space": self.action_space,
            "compare_role": self.compare_role,
            "default_checkpoint": self.default_checkpoint,
            "action_preview": self.action_preview,
            "error": self.error or "",
        }


def _first_action(result: VLAAction | VLAActionChunk) -> VLAAction:
    if isinstance(result, VLAActionChunk):
        return result.actions[0]
    return result


def _maybe_release_gpu() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compare_models(
    models: list[str],
    observation: VLAObservation,
    *,
    device: str = "auto",
    pretrained_overrides: dict[str, str] | None = None,
    load_kwargs: dict[str, dict[str, Any]] | None = None,
) -> list[ModelCompareResult]:
    """Run the same observation through each adapter sequentially.

    Models load one at a time so large checkpoints do not compete for GPU memory.
    """

    if not models:
        return []

    overrides = pretrained_overrides or {}
    per_model_kwargs = dict(DEFAULT_LOAD_KWARGS)
    if load_kwargs:
        for name, kwargs in load_kwargs.items():
            merged = dict(per_model_kwargs.get(name, {}))
            merged.update(kwargs)
            per_model_kwargs[name] = merged

    results: list[ModelCompareResult] = []
    for name in models:
        try:
            info = get_adapter_info(name)
        except UnknownModelError as exc:
            results.append(ModelCompareResult(name=name, ok=False, error=str(exc)))
            continue

        meta = dict(info.metadata)
        kwargs = dict(per_model_kwargs.get(info.name, {}))
        kwargs.setdefault("device", device)
        if info.name in overrides:
            kwargs["pretrained"] = overrides[info.name]

        try:
            load_start = perf_counter()
            runtime = load_model(info.name, **kwargs)
            load_ms = (perf_counter() - load_start) * 1000.0

            infer_start = perf_counter()
            raw = runtime.predict(observation=observation)
            infer_ms = (perf_counter() - infer_start) * 1000.0
            action = _first_action(raw)

            names = action.spec.names or tuple(f"a{i}" for i in range(action.data.size))
            values = tuple(float(x) for x in action.data.reshape(-1))
            results.append(
                ModelCompareResult(
                    name=info.name,
                    ok=True,
                    load_ms=load_ms,
                    infer_ms=infer_ms,
                    action_space=action.spec.action_space,
                    action_shape=tuple(int(x) for x in action.spec.shape),
                    action_names=tuple(str(n) for n in names),
                    action_values=values,
                    compare_role=str(meta.get("compare_role", "")),
                    default_checkpoint=str(meta.get("default_checkpoint", "")),
                    experimental=info.experimental,
                    metadata=dict(action.metadata),
                )
            )
        except VLARos2Error as exc:
            results.append(
                ModelCompareResult(
                    name=info.name,
                    ok=False,
                    error=str(exc),
                    compare_role=str(meta.get("compare_role", "")),
                    default_checkpoint=str(meta.get("default_checkpoint", "")),
                    experimental=info.experimental,
                )
            )
        except Exception as exc:  # noqa: BLE001 — collect per-model failures
            results.append(
                ModelCompareResult(
                    name=info.name,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    compare_role=str(meta.get("compare_role", "")),
                    default_checkpoint=str(meta.get("default_checkpoint", "")),
                    experimental=info.experimental,
                )
            )
        finally:
            _maybe_release_gpu()

    return results


def compare_results_to_rows(results: list[ModelCompareResult]) -> list[dict[str, Any]]:
    return [item.to_row() for item in results]


def compare_results_to_json(results: list[ModelCompareResult], *, indent: int = 2) -> str:
    payload = [asdict(item) for item in results]
    return json.dumps(payload, indent=indent)
