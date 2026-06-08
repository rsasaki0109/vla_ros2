#!/usr/bin/env python3
"""Browser playground for vla_ros2 adapters.

Upload an image (optional), enter a task instruction, pick an adapter, and inspect
the predicted action vector.

Run:

    pip install -e ".[playground,smolvla]"   # smolvla optional
    python scripts/vla_playground.py

Open http://127.0.0.1:7860
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from vla_ros2 import list_models, load_model
from vla_ros2.core.errors import VLARos2Error
from vla_ros2.core.types import VLAAction, VLAActionChunk, VLAObservation

DEFAULT_INSTRUCTION = "stack the red block on the blue block"
DEFAULT_MODEL = "dummy"


def _adapter_choices() -> list[str]:
    return [info.name for info in list_models()]


def _format_action(action: VLAAction) -> str:
    payload = {
        "action_space": action.spec.action_space,
        "shape": list(action.spec.shape),
        "data": action.data.reshape(-1).tolist(),
        "metadata": action.metadata,
    }
    return json.dumps(payload, indent=2)


def _action_bars(action: VLAAction) -> dict[str, float] | None:
    values = action.data.reshape(-1).astype(float)
    if values.size == 0:
        return None
    names = action.spec.names or tuple(f"a{i}" for i in range(values.size))
    return {str(name): float(value) for name, value in zip(names, values, strict=False)}


def predict(
    model_name: str,
    instruction: str,
    image: Any,
    device: str,
    pretrained: str,
    state_json: str,
) -> tuple[str, dict[str, float] | None, str]:
    if not model_name:
        return "Select a model.", None, ""

    state: dict[str, Any] = {}
    if state_json.strip():
        parsed = json.loads(state_json)
        if not isinstance(parsed, dict):
            msg = "state JSON must be an object"
            raise ValueError(msg)
        state = parsed

    images: dict[str, Any] = {}
    if image is not None:
        images["camera1"] = image

    observation = VLAObservation(
        instruction=instruction or DEFAULT_INSTRUCTION,
        images=images,
        state=state,
        metadata={"source": "vla_playground"},
    )

    kwargs: dict[str, Any] = {"device": device}
    if pretrained.strip():
        kwargs["pretrained"] = pretrained.strip()

    try:
        runtime = load_model(model_name, **kwargs)
        result = runtime.predict(observation=observation)
    except VLARos2Error as exc:
        return f"error: {exc}", None, ""
    except Exception as exc:  # noqa: BLE001 — surface to UI
        return f"error: {type(exc).__name__}: {exc}", None, ""

    if isinstance(result, VLAActionChunk):
        first = result.actions[0]
        summary = (
            f"chunk ({len(result.actions)} steps); showing step 0\n"
            f"{_format_action(first)}"
        )
        return summary, _action_bars(first), json.dumps(result.metadata, indent=2)

    if isinstance(result, VLAAction):
        return _format_action(result), _action_bars(result), json.dumps(result.metadata, indent=2)

    return f"unexpected result type: {type(result).__name__}", None, ""


def build_app() -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        msg = 'Playground requires gradio: pip install -e ".[playground]"'
        raise RuntimeError(msg) from exc

    models = _adapter_choices()
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]

    with gr.Blocks(title="vla_ros2 Playground") as demo:
        gr.Markdown(
            "# vla_ros2 Playground\n"
            "Try adapters locally before wiring them into ROS2. "
            "GPU models (SmolVLA, OpenVLA, Pi0) need their optional extras installed."
        )
        with gr.Row():
            model = gr.Dropdown(models, value=default_model, label="Adapter")
            device = gr.Dropdown(
                ["auto", "cuda", "cpu"],
                value="auto",
                label="Device",
            )
        instruction = gr.Textbox(
            value=DEFAULT_INSTRUCTION,
            label="Instruction",
            lines=2,
        )
        image = gr.Image(type="pil", label="Image (optional; camera1)")
        pretrained = gr.Textbox(
            label="Checkpoint override (optional)",
            placeholder="lerobot/smolvla_base or checkpoints/.../pretrained_model",
        )
        state_json = gr.Textbox(
            label="Proprio state JSON (optional)",
            placeholder='{"state": [0, 45, 90, 0, 0, 0.5]}',
            lines=2,
        )
        run_btn = gr.Button("Predict", variant="primary")
        action_json = gr.Code(label="Action JSON", language="json")
        action_plot = gr.Label(label="Action components")
        meta_json = gr.Code(label="Metadata", language="json")

        run_btn.click(
            predict,
            inputs=[model, instruction, image, device, pretrained, state_json],
            outputs=[action_json, action_plot, meta_json],
        )

    return demo


def main() -> None:
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
