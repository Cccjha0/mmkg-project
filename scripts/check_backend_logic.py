from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    backend_path = repo_root / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    from app.main import app

    client = TestClient(app)

    get_paths = [
        "/api/health",
        "/api/meta/runtime",
        "/api/meta/model",
        "/api/performance/overview",
        "/api/performance/accuracy-curves?dataset=OpenBG-IMG",
        "/api/performance/model-comparison?dataset=OpenBG-IMG",
        "/api/entities/ent_007314",
        "/api/entities/ent_007314/attribute-completion?topk=3",
        "/api/graph/subgraph?entity=ent_007314&hops=1&limit=10",
    ]

    for path in get_paths:
        response = client.get(path)
        assert response.status_code == 200, f"{path} failed: {response.status_code} {response.text}"

    predict_response = client.post(
        "/api/predict/tail",
        json={"head": "ent_007314", "relation": "rel_0001", "topk": 3},
    )
    assert (
        predict_response.status_code == 200
    ), f"/api/predict/tail failed: {predict_response.status_code} {predict_response.text}"

    payload = predict_response.json()
    assert payload["task"] == "tail_prediction"
    assert len(payload["results"]) == 3

    print("Backend logic check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
