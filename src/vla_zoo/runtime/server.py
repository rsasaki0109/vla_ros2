# mypy: disable-error-code=untyped-decorator
from __future__ import annotations

from time import perf_counter
from typing import Any

from vla_zoo.core.errors import MissingDependencyError
from vla_zoo.core.image import decode_image_base64
from vla_zoo.core.registry import list_models, load_model
from vla_zoo.runtime.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    prediction_to_response,
    request_to_observation,
)


def _import_fastapi() -> tuple[Any, Any]:
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        msg = 'Inference server requires FastAPI. Install with: pip install "vla_zoo[server]"'
        raise MissingDependencyError(msg) from exc
    return FastAPI, HTTPException


def create_app(model_name: str = "dummy", **model_kwargs: Any) -> Any:
    """Create a FastAPI app for VLA inference."""

    FastAPI, HTTPException = _import_fastapi()
    model = load_model(model_name, runtime="local", **model_kwargs)
    app = FastAPI(
        title="vla_zoo inference server",
        version="0.1.0",
        description="HTTP runtime for Vision-Language-Action model adapters.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(ready=True, model=model.name, runtime="server")

    @app.get("/v1/models", response_model=list[ModelInfoResponse])
    def models() -> list[ModelInfoResponse]:
        return [
            ModelInfoResponse(
                name=info.name,
                source=info.source,
                aliases=list(info.aliases),
                experimental=info.experimental,
                domain=info.domain,
                description=info.description,
            )
            for info in list_models()
        ]

    @app.post("/v1/predict")
    def predict(request: PredictRequest) -> Any:
        if request.model != model.name and request.model != model.model_id:
            raise HTTPException(
                status_code=400,
                detail=f"Server hosts {model.name!r}; request asked for {request.model!r}",
            )
        try:
            images = {
                name: decode_image_base64(payload.model_dump(mode="json"))
                for name, payload in request.images.items()
            }
            observation = request_to_observation(request, images)
            start = perf_counter()
            action = model.predict(observation=observation)
            latency_ms = (perf_counter() - start) * 1000.0
            if hasattr(action, "metadata"):
                action.metadata.setdefault("latency_ms", latency_ms)
                action.metadata.setdefault("model", model.name)
            response = prediction_to_response(action)
            return response.model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/reset")
    def reset() -> dict[str, str]:
        return {"status": "ok"}

    return app


def run_server(model_name: str, host: str, port: int, **model_kwargs: Any) -> None:
    """Run the HTTP inference server via uvicorn."""

    try:
        import uvicorn
    except ImportError as exc:
        msg = 'Serving requires uvicorn. Install with: pip install "vla_zoo[server]"'
        raise MissingDependencyError(msg) from exc
    app = create_app(model_name=model_name, **model_kwargs)
    uvicorn.run(app, host=host, port=port)
