from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.services import attribute_service, entity_service


client = TestClient(app)


def _patch_entity_metadata(monkeypatch, *, has_image: bool = False) -> None:
    monkeypatch.setattr(entity_service, "entity_exists", lambda entity: entity == "ent_000001")
    monkeypatch.setattr(entity_service, "entity_text", lambda entity: "裤子" if entity == "ent_000001" else None)
    monkeypatch.setattr(entity_service, "entity_text_en", lambda entity: "Pants" if entity == "ent_000001" else None)
    monkeypatch.setattr(
        entity_service,
        "image_path_for_entity",
        lambda entity: "/static/openbg_img/ent_000001/image_0.jpg" if has_image else None,
    )
    monkeypatch.setattr(entity_service, "entity_embedding_summary", lambda entity: {"text": {"dim": 256, "l2_norm": 1.0}})


def test_entity_detail_returns_metadata_only_when_model_is_missing(monkeypatch) -> None:
    _patch_entity_metadata(monkeypatch, has_image=False)
    monkeypatch.setattr(entity_service, "predictor_status", lambda: {"model_ready": False})

    response = client.get("/api/entities/ent_000001")

    assert response.status_code == 200
    payload = response.json()
    assert payload["entity"] == "ent_000001"
    assert payload["entity_text_zh"] == "裤子"
    assert payload["entity_text_en"] == "Pants"
    assert payload["has_image"] is False
    assert payload["image_status"] == "missing"
    assert payload["available_spaces"] == ["text"]
    assert payload["model_name"].endswith("(metadata only)")


def test_entity_detail_marks_image_available(monkeypatch) -> None:
    _patch_entity_metadata(monkeypatch, has_image=True)
    monkeypatch.setattr(entity_service, "predictor_status", lambda: {"model_ready": False})

    response = client.get("/api/entities/ent_000001")

    assert response.status_code == 200
    payload = response.json()
    assert payload["has_image"] is True
    assert payload["image_status"] == "available"
    assert payload["image_path"] == "/static/openbg_img/ent_000001/image_0.jpg"


def test_entity_detail_returns_404_for_unknown_entity(monkeypatch) -> None:
    monkeypatch.setattr(entity_service, "entity_exists", lambda entity: False)

    response = client.get("/api/entities/ent_999999")

    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "ENTITY_NOT_FOUND"


def test_similar_entities_returns_503_when_model_is_missing(monkeypatch) -> None:
    _patch_entity_metadata(monkeypatch)
    monkeypatch.setattr(
        entity_service,
        "predictor_status",
        lambda: {
            "model_ready": False,
            "run_dir": "missing",
            "config_path": "missing/config_merged.json",
            "checkpoint_path": "missing/best.ckpt",
            "load_error": None,
        },
    )

    response = client.get("/api/entities/ent_000001/similar")

    assert response.status_code == 503
    assert response.json()["detail"]["error"]["code"] == "MODEL_NOT_READY"


def test_attribute_completion_returns_existing_rows_when_model_is_missing(monkeypatch) -> None:
    _patch_entity_metadata(monkeypatch, has_image=True)
    monkeypatch.setattr(attribute_service, "predictor_status", lambda: {"model_ready": False})
    monkeypatch.setattr(attribute_service, "entity_text", lambda entity: {"ent_000001": "裤子", "ent_000002": "黑色"}.get(entity))
    monkeypatch.setattr(attribute_service, "entity_text_en", lambda entity: {"ent_000001": "Pants", "ent_000002": "Black"}.get(entity))
    monkeypatch.setattr(attribute_service, "image_path_for_entity", lambda entity: "/static/openbg_img/ent_000001/image_0.jpg")
    monkeypatch.setattr(attribute_service, "relation_text", lambda relation: "颜色")
    monkeypatch.setattr(attribute_service, "relation_text_en", lambda relation: "color")
    monkeypatch.setattr(attribute_service, "triples_by_pair", lambda: {("ent_000001", "rel_0001"): ["ent_000002"]})

    response = client.get("/api/entities/ent_000001/attribute-completion", params={"topk": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "attribute_completion_page"
    assert payload["results"]["entity_info"]["has_image"] is True
    rows = payload["results"]["attribute_rows"]
    assert rows[0]["source"] == "existing"
    assert rows[0]["relation_name"] == "颜色"
    assert rows[0]["relation_name_en"] == "color"
    assert rows[0]["selected_value"]["entity_text_en"] == "Black"
    assert rows[0]["warning"].startswith("Model artifacts are not ready")


def test_attribute_completion_validates_topk_upper_bound() -> None:
    response = client.get("/api/entities/ent_000001/attribute-completion", params={"topk": 21})

    assert response.status_code == 422
