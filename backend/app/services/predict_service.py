from __future__ import annotations

from time import perf_counter

from fastapi import HTTPException

from app.deps import get_runtime_config
from app.schemas.predict import TailPredictionCandidate, TailPredictionRequest, TailPredictionResponse
from app.services.entity_service import _ensure_entity
from app.services.inference_service import get_predictor, predictor_status, score_to_display, score_to_normalized
from app.services.openbg_img_data import entity_text, entity_text_en, relation_exists, triples_by_pair


def _ensure_relation(relation: str) -> None:
    if relation_exists(relation):
        return
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "code": "RELATION_NOT_FOUND",
                "message": f"Relation {relation} not found in OpenBG-IMG.",
                "details": None,
            }
        },
    )


def _candidate(entity: str, *, raw_score: float, normalized_score: float, rank: int) -> TailPredictionCandidate:
    return TailPredictionCandidate(
        entity=entity,
        entity_text_zh=entity_text(entity),
        entity_text_en=entity_text_en(entity),
        raw_score=float(raw_score),
        normalized_score=float(normalized_score),
        display_score=score_to_display(normalized_score),
        rank=rank,
    )


def predict_tail(request: TailPredictionRequest) -> TailPredictionResponse:
    start = perf_counter()
    cfg = get_runtime_config()
    _ensure_entity(request.head)
    _ensure_relation(request.relation)

    observed_tails = triples_by_pair().get((request.head, request.relation), [])
    if observed_tails:
        results = [
            _candidate(entity=tail, raw_score=1.0, normalized_score=1.0 if idx == 0 else 0.0, rank=idx + 1)
            for idx, tail in enumerate(observed_tails[: request.topk])
        ]
        return TailPredictionResponse(
            model=cfg["model_name"],
            device=cfg["device"],
            inputs=request,
            results=results,
            latency_ms=round((perf_counter() - start) * 1000, 3),
        )

    status = predictor_status()
    if not status["model_ready"]:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "MODEL_NOT_READY",
                    "message": "Model artifacts are not ready for tail prediction.",
                    "details": status,
                }
            },
        )

    predictor = get_predictor()
    prediction = predictor.predict_tail(
        head_id=request.head,
        rel_id=request.relation,
        topk=request.topk,
    )

    raw_scores = [float(item["score"]) for item in prediction["results"]]
    normalized_scores = score_to_normalized(raw_scores)
    results = [
        _candidate(
            entity=item["entity"],
            raw_score=raw_scores[idx],
            normalized_score=normalized_scores[idx],
            rank=idx + 1,
        )
        for idx, item in enumerate(prediction["results"])
    ]

    return TailPredictionResponse(
        model=cfg["model_name"],
        device=cfg["device"],
        inputs=request,
        results=results,
        latency_ms=round((perf_counter() - start) * 1000, 3),
    )
