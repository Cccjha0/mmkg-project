from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint_returns_stable_shape() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] == "OpenBG-IMG"
    assert "model_loaded" in payload
    assert "run_dir" in payload
    assert isinstance(payload["warnings"], list)


def test_performance_rejects_unknown_dataset() -> None:
    response = client.get("/api/performance/accuracy-curves", params={"dataset": "NotARealDataset"})

    assert response.status_code == 422


def test_graph_query_limits_are_validated_before_expansion() -> None:
    response = client.get("/api/graph/subgraph", params={"entity": "ent_000001", "hops": 3, "limit": 101})

    assert response.status_code == 422
