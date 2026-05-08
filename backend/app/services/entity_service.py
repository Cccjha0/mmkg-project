from __future__ import annotations

from time import perf_counter

from fastapi import HTTPException

from app.deps import get_runtime_config
from app.schemas.entity import EmbeddingSummary, EntityInfoResponse, SimilarEntitiesResponse, SimilarEntityItem
from app.services.inference_service import get_predictor, predictor_status
from app.services.openbg_img_data import entity_exists, entity_text, entity_text_en, image_path_for_entity


def _raise_entity_not_found(entity_id: str) -> None:
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "code": "ENTITY_NOT_FOUND",
                "message": f"Entity {entity_id} not found in OpenBG-IMG.",
                "details": None,
            }
        },
    )


def _ensure_entity(entity_id: str) -> None:
    if not entity_exists(entity_id):
        _raise_entity_not_found(entity_id)


def get_entity_info(entity_id: str) -> EntityInfoResponse:
    cfg = get_runtime_config()
    _ensure_entity(entity_id)
    status = predictor_status()
    if not status["model_ready"]:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "MODEL_NOT_READY",
                    "message": "Model artifacts are not ready for entity inspection.",
                    "details": status,
                }
            },
        )

    predictor = get_predictor()
    entity = predictor.get_entity_multimodal(entity_id)
    result = entity["results"]

    summary = {
        key: EmbeddingSummary(**value)
        for key, value in result.get("embedding_summary", {}).items()
    }
    gate_summary = result.get("gate_summary") or {}

    return EntityInfoResponse(
        entity=result["entity"],
        entity_text=result.get("entity_text"),
        entity_text_zh=result.get("entity_text_zh"),
        entity_text_en=result.get("entity_text_en"),
        has_image=bool(result.get("has_image")),
        image_path=result.get("image_path"),
        image_status="available" if result.get("image_path") else "missing",
        available_spaces=result.get("available_spaces", []),
        embedding_summary=summary,
        gate_mean=gate_summary.get("mean_gate"),
        model_name=cfg["model_name"],
        dataset_name="openbg_img",
    )


def get_similar_entities(entity_id: str, space: str, topk: int) -> SimilarEntitiesResponse:
    start = perf_counter()
    cfg = get_runtime_config()
    _ensure_entity(entity_id)
    status = predictor_status()
    if not status["model_ready"]:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "MODEL_NOT_READY",
                    "message": "Model artifacts are not ready for similarity search.",
                    "details": status,
                }
            },
        )

    predictor = get_predictor()
    result = predictor.similar_entities(entity_id=entity_id, topk=topk, space=space)

    items = [
        SimilarEntityItem(
            entity=item["entity"],
            entity_text_zh=item.get("entity_text_zh"),
            entity_text_en=item.get("entity_text_en"),
            score=max(0.0, min(1.0, float(item["score"]))),
        )
        for item in result["results"]
    ]

    latency_ms = round((perf_counter() - start) * 1000, 3)
    return SimilarEntitiesResponse(
        model=cfg["model_name"],
        device=cfg["device"],
        inputs=result["inputs"],
        results=items,
        latency_ms=latency_ms,
    )
