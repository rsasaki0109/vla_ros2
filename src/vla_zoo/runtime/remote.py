from __future__ import annotations

from typing import Any

from vla_zoo.core.errors import MissingDependencyError, RemoteRuntimeError
from vla_zoo.core.image import encode_image_base64
from vla_zoo.core.model import BaseVLA
from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation
from vla_zoo.runtime.schemas import (
    ActionChunkResponse,
    ActionResponse,
    ImagePayload,
    PredictRequest,
    response_to_prediction,
)

REMOTE_PLACEHOLDER_SPEC = ActionSpec(
    action_space="custom",
    shape=(1,),
    description="Remote placeholder action spec; responses carry the concrete spec.",
)


class RemoteVLAClient(BaseVLA):
    """HTTP client implementing the same prediction interface as local adapters."""

    name = "remote"
    model_id = "remote"
    action_spec = REMOTE_PLACEHOLDER_SPEC

    def __init__(
        self,
        *,
        model_name: str,
        remote_url: str,
        timeout: float = 30.0,
        **metadata: Any,
    ) -> None:
        super().__init__(
            name=model_name,
            model_id=model_name,
            action_spec=REMOTE_PLACEHOLDER_SPEC,
            metadata=metadata,
        )
        try:
            import httpx
        except ImportError as exc:
            msg = 'Remote runtime requires httpx. Install with: pip install "vla_zoo[server]"'
            raise MissingDependencyError(msg) from exc
        self._httpx = httpx
        self.model_name = model_name
        self.remote_url = remote_url.rstrip("/")
        self.timeout = timeout

    def _encode_images(self, observation: VLAObservation) -> dict[str, ImagePayload]:
        encoded: dict[str, ImagePayload] = {}
        for name, image in observation.images.items():
            if isinstance(image, dict) and "encoding" in image and "data" in image:
                encoded[name] = ImagePayload.model_validate(image)
            else:
                encoded[name] = ImagePayload.model_validate(encode_image_base64(image))
        return encoded

    def predict_observation(self, observation: VLAObservation) -> VLAAction | VLAActionChunk:
        request = PredictRequest(
            model=self.model_name,
            instruction=observation.instruction,
            images=self._encode_images(observation),
            state=dict(observation.state),
            metadata=observation.metadata,
        )
        try:
            with self._httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.remote_url}/v1/predict",
                    json=request.model_dump(mode="json"),
                )
                response.raise_for_status()
        except Exception as exc:
            msg = f"Remote VLA prediction failed for {self.remote_url}: {exc}"
            raise RemoteRuntimeError(msg) from exc

        payload = response.json()
        if payload.get("kind") == "chunk":
            parsed: ActionResponse | ActionChunkResponse = ActionChunkResponse.model_validate(
                payload
            )
        else:
            parsed = ActionResponse.model_validate(payload)
        prediction = response_to_prediction(parsed)
        if isinstance(prediction, VLAAction):
            self.action_spec = prediction.spec
        elif prediction.actions:
            self.action_spec = prediction.actions[0].spec
        return prediction
