from __future__ import annotations

from fastapi.testclient import TestClient

from vla_zoo.runtime.server import create_app


def test_server_health_and_models() -> None:
    client = TestClient(create_app("dummy"))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "ready": True,
        "model": "dummy",
        "runtime": "server",
        "status": "ok",
    }

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert "dummy" in {item["name"] for item in models.json()}


def test_server_predict_dummy() -> None:
    client = TestClient(create_app("dummy"))

    response = client.post(
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
    client = TestClient(create_app("dummy"))

    response = client.post(
        "/v1/predict",
        json={"model": "openvla", "instruction": "test"},
    )

    assert response.status_code == 400
