from __future__ import annotations

from typing import Any

import anyio
import httpx

from vla_zoo.runtime.server import create_app


def _request(method: str, path: str, *, json: dict[str, Any] | None = None) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app("dummy"))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, json=json)

    return anyio.run(run)


def test_server_health_and_models() -> None:
    health = _request("GET", "/health")
    assert health.status_code == 200
    assert health.json() == {
        "ready": True,
        "model": "dummy",
        "runtime": "server",
        "status": "ok",
    }

    models = _request("GET", "/v1/models")
    assert models.status_code == 200
    assert "dummy" in {item["name"] for item in models.json()}


def test_server_predict_dummy() -> None:
    response = _request(
        "POST",
        "/v1/predict",
        json={
            "model": "dummy",
            "instruction": "test",
            "images": {},
            "state": {},
            "metadata": {"source": "test"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action_space"] == "eef_delta"
    assert payload["data"] == [0.0] * 7
    assert payload["metadata"]["model"] == "dummy"
    assert "latency_ms" in payload["metadata"]


def test_server_rejects_wrong_model() -> None:
    response = _request(
        "POST",
        "/v1/predict",
        json={"model": "openvla", "instruction": "test"},
    )

    assert response.status_code == 400
