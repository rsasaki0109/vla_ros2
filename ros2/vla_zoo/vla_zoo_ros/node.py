from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from time import perf_counter
from typing import Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from vla_zoo_msgs.msg import VLAAction as VLAActionMsg
from vla_zoo_msgs.msg import VLAActionChunk as VLAActionChunkMsg
from vla_zoo_msgs.msg import VLAStatus as VLAStatusMsg

from vla_zoo import load_model
from vla_zoo.core.types import VLAAction, VLAActionChunk, VLAObservation
from vla_zoo_ros.converters import action_to_msg, chunk_to_msg, ros_image_to_numpy, status_to_msg
from vla_zoo_ros.params import RuntimeNodeParams
from vla_zoo_ros.qos import action_qos, image_qos, instruction_qos, status_qos


class VLARuntimeNode(Node):
    def __init__(self) -> None:
        super().__init__("vla_runtime_node")
        self._declare_parameters()
        self.params = self._read_params()
        self._latest_image: Image | None = None
        self._latest_instruction: str | None = None
        self._latest_joint_state: JointState | None = None
        self._pending: Future[VLAAction | VLAActionChunk] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._latencies_ms: list[float] = []
        self._dropped_frames = 0
        self._status_text = "starting"

        model_kwargs: dict[str, Any] = {
            "device": self.params.device,
            "pretrained": self.params.pretrained,
            "unnorm_key": self.params.unnorm_key,
        }
        if self.params.runtime == "remote":
            model_kwargs["remote_url"] = self.params.remote_url
        if self.params.model_name == "dummy":
            model_kwargs = {}
        self.model = load_model(self.params.model_name, runtime=self.params.runtime, **model_kwargs)

        self.create_subscription(Image, self.params.image_topic, self._image_cb, image_qos())
        self.create_subscription(
            String,
            self.params.instruction_topic,
            self._instruction_cb,
            instruction_qos(self.params.max_queue_size),
        )
        self.create_subscription(
            JointState,
            self.params.joint_state_topic,
            self._joint_state_cb,
            action_qos(self.params.max_queue_size),
        )
        self.action_pub = self.create_publisher(
            VLAActionMsg,
            self.params.action_topic,
            action_qos(self.params.max_queue_size),
        )
        self.chunk_pub = self.create_publisher(
            VLAActionChunkMsg,
            self.params.action_chunk_topic,
            action_qos(self.params.max_queue_size),
        )
        self.status_pub = self.create_publisher(
            VLAStatusMsg,
            self.params.status_topic,
            status_qos(self.params.max_queue_size),
        )
        self.timer = self.create_timer(1.0 / self.params.control_hz, self._tick)
        self._status_text = "ready"
        self.get_logger().info(
            f"vla_runtime_node ready: model={self.params.model_name} "
            f"runtime={self.params.runtime} dry_run={self.params.dry_run}"
        )

    def _declare_parameters(self) -> None:
        defaults: dict[str, Any] = {
            "model_name": "dummy",
            "runtime": "local",
            "dry_run": True,
            "image_topic": "/camera/image_raw",
            "instruction_topic": "/vla/instruction",
            "joint_state_topic": "/joint_states",
            "action_topic": "/vla/action",
            "action_chunk_topic": "/vla/action_chunk",
            "status_topic": "/vla/status",
            "publish_action_chunk": True,
            "control_hz": 5.0,
            "max_queue_size": 1,
            "device": "cuda:0",
            "pretrained": "openvla/openvla-7b",
            "unnorm_key": "bridge_orig",
            "remote_url": "http://localhost:8000",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_params(self) -> RuntimeNodeParams:
        def value(name: str) -> Any:
            return self.get_parameter(name).value

        def as_bool(name: str) -> bool:
            raw = value(name)
            if isinstance(raw, str):
                return raw.lower() in {"1", "true", "yes", "on"}
            return bool(raw)

        return RuntimeNodeParams(
            model_name=str(value("model_name")),
            runtime=str(value("runtime")),
            dry_run=as_bool("dry_run"),
            image_topic=str(value("image_topic")),
            instruction_topic=str(value("instruction_topic")),
            joint_state_topic=str(value("joint_state_topic")),
            action_topic=str(value("action_topic")),
            action_chunk_topic=str(value("action_chunk_topic")),
            status_topic=str(value("status_topic")),
            publish_action_chunk=as_bool("publish_action_chunk"),
            control_hz=float(value("control_hz")),
            max_queue_size=int(value("max_queue_size")),
            device=str(value("device")),
            pretrained=str(value("pretrained")),
            unnorm_key=str(value("unnorm_key")),
            remote_url=str(value("remote_url")),
        )

    def _image_cb(self, msg: Image) -> None:
        self._latest_image = msg

    def _instruction_cb(self, msg: String) -> None:
        self._latest_instruction = msg.data

    def _joint_state_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _tick(self) -> None:
        self._publish_status()
        if self._pending is not None:
            if self._pending.done():
                future = self._pending
                self._pending = None
                self._handle_prediction(future)
            return
        if self._latest_instruction is None:
            self._status_text = "waiting for instruction"
            return
        try:
            observation = self._make_observation()
        except Exception as exc:
            self._status_text = f"waiting for valid observation: {exc}"
            return
        self._pending = self._executor.submit(self._predict_timed, observation)

    def _make_observation(self) -> VLAObservation:
        images: dict[str, Any] = {}
        if self._latest_image is not None:
            images["primary"] = ros_image_to_numpy(self._latest_image)
        state: dict[str, Any] = {}
        if self._latest_joint_state is not None:
            state["joint_names"] = list(self._latest_joint_state.name)
            state["joint_position"] = list(self._latest_joint_state.position)
            state["joint_velocity"] = list(self._latest_joint_state.velocity)
        return VLAObservation(
            instruction=self._latest_instruction or "",
            images=images,
            state=state,
            metadata={"dry_run": self.params.dry_run},
        )

    def _predict_timed(self, observation: VLAObservation) -> VLAAction | VLAActionChunk:
        start = perf_counter()
        action = self.model.predict(observation=observation)
        latency = (perf_counter() - start) * 1000.0
        self._latencies_ms.append(latency)
        if len(self._latencies_ms) > 100:
            self._latencies_ms.pop(0)
        return action

    def _handle_prediction(self, future: Future[VLAAction | VLAActionChunk]) -> None:
        try:
            prediction = future.result()
        except Exception as exc:
            self._status_text = f"inference error: {exc}"
            self.get_logger().error(self._status_text)
            return
        stamp = self.get_clock().now().to_msg()
        if isinstance(prediction, VLAActionChunk):
            if self.params.publish_action_chunk:
                self.chunk_pub.publish(
                    chunk_to_msg(
                        prediction,
                        stamp=stamp,
                        model_name=self.model.name,
                        adapter_name=type(self.model).__name__,
                    )
                )
            self.action_pub.publish(
                action_to_msg(
                    prediction.actions[0],
                    stamp=stamp,
                    model_name=self.model.name,
                    adapter_name=type(self.model).__name__,
                )
            )
        else:
            self.action_pub.publish(
                action_to_msg(
                    prediction,
                    stamp=stamp,
                    model_name=self.model.name,
                    adapter_name=type(self.model).__name__,
                )
            )
        self._status_text = "ready"

    def _publish_status(self) -> None:
        stamp = self.get_clock().now().to_msg()
        last_latency = self._latencies_ms[-1] if self._latencies_ms else 0.0
        avg_latency = (
            sum(self._latencies_ms) / len(self._latencies_ms) if self._latencies_ms else 0.0
        )
        self.status_pub.publish(
            status_to_msg(
                stamp=stamp,
                model_name=getattr(self, "model", None).name if hasattr(self, "model") else "",
                adapter_name=type(getattr(self, "model", object())).__name__,
                ready=self._status_text == "ready",
                dry_run=self.params.dry_run,
                last_latency_ms=last_latency,
                avg_latency_ms=avg_latency,
                action_rate_hz=self.params.control_hz,
                dropped_frames=self._dropped_frames,
                status_text=self._status_text,
            )
        )

    def destroy_node(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VLARuntimeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
