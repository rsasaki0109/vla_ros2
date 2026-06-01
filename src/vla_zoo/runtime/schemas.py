from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation


class ImagePayload(BaseModel):
    encoding: str
    data: str


class PredictRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    instruction: str
    images: dict[str, ImagePayload] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionResponse(BaseModel):
    action_space: str
    data: list[float]
    shape: list[int]
    names: list[str] = Field(default_factory=list)
    frame_id: str | None = None
    control_hz: float | None = None
    normalized: bool = False
    dt: float | None = None
    confidence: float | None = None
    chunk_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionChunkResponse(BaseModel):
    kind: Literal["chunk"] = "chunk"
    actions: list[ActionResponse]
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    ready: bool
    model: str
    runtime: str
    status: str = "ok"


class ModelInfoResponse(BaseModel):
    name: str
    source: str
    aliases: list[str] = Field(default_factory=list)
    experimental: bool = False
    domain: str | None = None
    description: str = ""


def action_to_response(action: VLAAction) -> ActionResponse:
    return ActionResponse(
        action_space=action.spec.action_space,
        data=action.tolist(),
        shape=list(action.spec.shape),
        names=list(action.spec.names),
        frame_id=action.spec.frame_id,
        control_hz=action.spec.control_hz,
        normalized=action.spec.normalized,
        dt=action.dt,
        confidence=action.confidence,
        chunk_index=action.chunk_index,
        metadata=action.metadata,
    )


def prediction_to_response(
    action: VLAAction | VLAActionChunk,
) -> ActionResponse | ActionChunkResponse:
    if isinstance(action, VLAActionChunk):
        return ActionChunkResponse(
            actions=[action_to_response(item) for item in action.actions],
            metadata=action.metadata,
        )
    return action_to_response(action)


def response_to_action(response: ActionResponse) -> VLAAction:
    spec = ActionSpec(
        action_space=response.action_space,  # type: ignore[arg-type]
        shape=tuple(response.shape),
        names=tuple(response.names),
        frame_id=response.frame_id,
        control_hz=response.control_hz,
        normalized=response.normalized,
    )
    import numpy as np

    return VLAAction(
        data=np.asarray(response.data, dtype=np.float32).reshape(spec.shape),
        spec=spec,
        dt=response.dt,
        confidence=response.confidence,
        chunk_index=response.chunk_index,
        metadata=response.metadata,
    )


def response_to_prediction(
    response: ActionResponse | ActionChunkResponse,
) -> VLAAction | VLAActionChunk:
    if isinstance(response, ActionChunkResponse):
        return VLAActionChunk(
            actions=[response_to_action(action) for action in response.actions],
            metadata=response.metadata,
        )
    return response_to_action(response)


def request_to_observation(request: PredictRequest, images: dict[str, Any]) -> VLAObservation:
    return VLAObservation(
        instruction=request.instruction,
        images=images,
        state=request.state,
        metadata={**request.metadata, "requested_model": request.model},
    )
