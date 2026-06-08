#!/usr/bin/env python3
"""Browser playground for vla_ros2 adapters.

Local mode: upload image + instruction, run adapter inference offline.

ROS2 mode (--ros): publish instructions to a live graph (e.g. Gazebo SmolVLA loop)
and display /vla/action, /vla/status, and /camera/image_raw.

Run local:

    pip install -e ".[playground,smolvla]"
    python scripts/vla_playground.py

Run with Gazebo (two terminals, no real robot):

    ./scripts/launch_playground_gz.sh
    python scripts/vla_playground.py --ros

Open http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from vla_ros2 import list_models, load_model
from vla_ros2.core.errors import VLARos2Error
from vla_ros2.core.types import VLAAction, VLAActionChunk, VLAObservation

DEFAULT_INSTRUCTION = "stack the red block on the blue block"
DEFAULT_MODEL = "dummy"
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _ros_bridge_cls() -> type:
    from vla_playground_ros_bridge import RosPlaygroundBridge

    return RosPlaygroundBridge


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


def build_compare_tab() -> None:
    import gradio as gr

    from vla_ros2.compare import compare_models, compare_results_to_json, compare_results_to_rows

    models = _adapter_choices()
    default_models = [name for name in ("dummy", "random", "scripted", "smolvla") if name in models]

    def _run_compare(
        selected: list[str],
        instruction: str,
        image: Any,
        device: str,
        pretrained_json: str,
        state_json: str,
    ) -> tuple[list[list[Any]], str, str]:
        if not selected:
            return [], "Select at least one adapter.", ""

        overrides: dict[str, str] = {}
        if pretrained_json.strip():
            parsed = json.loads(pretrained_json)
            if not isinstance(parsed, dict):
                msg = "pretrained overrides JSON must be an object"
                raise ValueError(msg)
            overrides = {str(k): str(v) for k, v in parsed.items()}

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
            metadata={"source": "vla_playground_compare"},
        )
        results = compare_models(
            selected,
            observation,
            device=device,
            pretrained_overrides=overrides,
        )
        rows = compare_results_to_rows(results)
        if not rows:
            return [], "No results.", ""

        columns = list(rows[0].keys())
        table = [[row[col] for col in columns] for row in rows]
        summary_lines = [
            f"- **{item.name}**: {'ok' if item.ok else item.error} "
            f"({item.infer_ms:.1f} ms infer)"
            if item.ok and item.infer_ms is not None
            else f"- **{item.name}**: {item.error or 'failed'}"
            for item in results
        ]
        return table, "\n".join(summary_lines), compare_results_to_json(results)

    with gr.Tab("Compare adapters"):
        gr.Markdown(
            "Run the **same** instruction/image/state through multiple adapters. "
            "Models load sequentially to reduce GPU memory pressure."
        )
        selected = gr.CheckboxGroup(models, value=default_models, label="Adapters")
        with gr.Row():
            device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
        instruction = gr.Textbox(value=DEFAULT_INSTRUCTION, label="Instruction", lines=2)
        image = gr.Image(type="pil", label="Image (optional; camera1)")
        state_json = gr.Textbox(
            label="Proprio state JSON (optional)",
            placeholder='{"state": [0, 45, 90, 0, 0, 0.5]}',
            lines=2,
        )
        pretrained_json = gr.Textbox(
            label="Checkpoint overrides JSON (optional)",
            placeholder='{"smolvla": "checkpoints/.../pretrained_model"}',
            lines=2,
        )
        compare_btn = gr.Button("Compare", variant="primary")
        compare_table = gr.Dataframe(
            headers=[
                "model",
                "ok",
                "load_ms",
                "infer_ms",
                "action_shape",
                "action_space",
                "compare_role",
                "default_checkpoint",
                "action_preview",
                "error",
            ],
            label="Comparison table",
            wrap=True,
        )
        compare_summary = gr.Markdown()
        compare_json = gr.Code(label="Full JSON", language="json")
        compare_btn.click(
            _run_compare,
            inputs=[selected, instruction, image, device, pretrained_json, state_json],
            outputs=[compare_table, compare_summary, compare_json],
        )


def build_local_tab() -> None:
    import gradio as gr

    models = _adapter_choices()
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]

    with gr.Tab("Local inference"):
        gr.Markdown(
            "Run adapters offline on this machine. "
            "GPU models need optional extras (`pip install -e \".[smolvla]\"`)."
        )
        with gr.Row():
            model = gr.Dropdown(models, value=default_model, label="Adapter")
            device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
        instruction = gr.Textbox(value=DEFAULT_INSTRUCTION, label="Instruction", lines=2)
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


def build_ros_tab(bridge: Any) -> None:
    import gradio as gr

    def _poll() -> tuple[Any, str, str, dict[str, float] | None]:
        snap = bridge.snapshot()
        if snap.error:
            return None, snap.error, snap.status_json, None
        return snap.camera, snap.action_json, snap.status_json, snap.action_bars

    def _send_instruction(text: str) -> str:
        bridge.publish_instruction(text or DEFAULT_INSTRUCTION, repeat=5)
        return f"published: {text or DEFAULT_INSTRUCTION}"

    with gr.Tab("ROS2 live (Gazebo / sim)"):
        gr.Markdown(
            "Connect to a running ROS2 graph. Start the sim stack first:\n\n"
            "`./scripts/launch_playground_gz.sh`"
        )
        ros_instruction = gr.Textbox(value=DEFAULT_INSTRUCTION, label="Instruction", lines=2)
        send_btn = gr.Button("Send to /vla/instruction", variant="primary")
        send_status = gr.Textbox(label="Publish status", interactive=False)
        ros_camera = gr.Image(type="pil", label="Live /camera/image_raw")
        ros_action = gr.Code(label="/vla/action", language="json")
        ros_status = gr.Code(label="/vla/status", language="json")
        ros_plot = gr.Label(label="Latest action")
        send_btn.click(_send_instruction, inputs=[ros_instruction], outputs=[send_status])
        gr.Timer(0.5).tick(
            fn=_poll,
            outputs=[ros_camera, ros_action, ros_status, ros_plot],
        )


def build_app(*, ros_mode: bool = False) -> tuple[Any, Any | None]:
    try:
        import gradio as gr
    except ImportError as exc:
        msg = 'Playground requires gradio: pip install -e ".[playground]"'
        raise RuntimeError(msg) from exc

    bridge = None
    if ros_mode:
        bridge = _ros_bridge_cls()()

    with gr.Blocks(title="vla_ros2 Playground") as demo:
        gr.Markdown("# vla_ros2 Playground")
        build_compare_tab()
        build_local_tab()
        if bridge is not None:
            build_ros_tab(bridge)
    return demo, bridge


def main() -> None:
    import atexit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ros",
        action="store_true",
        help="Enable ROS2 live tab (requires sourced ROS2 + running graph).",
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo, bridge = build_app(ros_mode=args.ros)
    if bridge is not None:
        atexit.register(bridge.close)
        atexit.register(bridge.shutdown_ros)
    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
