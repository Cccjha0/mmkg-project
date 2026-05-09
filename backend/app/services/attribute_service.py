from __future__ import annotations

import logging
from time import perf_counter

from fastapi import HTTPException

from app.deps import get_runtime_config
from app.schemas.attribute_completion import AttributeCandidate, AttributeCompletionResponse, AttributeRow
from app.services.entity_service import _ensure_entity
from app.services.inference_service import get_predictor, predictor_status, score_to_display, score_to_normalized
from app.services.openbg_img_data import (
    entity_text,
    entity_text_en,
    image_path_for_entity,
    relation_tail_candidates,
    relation_tail_count,
    relation_text,
    relation_text_en,
    triples_by_pair,
)

logger = logging.getLogger(__name__)


def _attribute_relations(predictor) -> list[str]:
    configured = getattr(predictor, "cfg", {}).get("inference", {}).get("attribute_relations")
    if not configured:
        raise RuntimeError("No inference.attribute_relations configured in model config.")
    return [predictor.format_relation(predictor.parse_relation(rel)) for rel in configured]


def _candidate(entity: str, raw_score: float | None, normalized_score: float | None, rank: int | None) -> AttributeCandidate:
    text_zh = entity_text(entity)
    text_en = entity_text_en(entity)
    display_score = None
    if normalized_score is not None:
        display_score = normalized_score if normalized_score >= 0.1 else round(normalized_score, 4)
    return AttributeCandidate(
        entity=entity,
        entity_text_zh=text_zh,
        entity_text_en=text_en,
        score=display_score,
        raw_score=raw_score,
        normalized_score=normalized_score,
        display_score=display_score,
        rank=rank,
    )


def _dedupe_options(options: list[AttributeCandidate], *, topk: int) -> list[AttributeCandidate]:
    grouped: dict[tuple[str, str], list[AttributeCandidate]] = {}
    for option in options:
        key = ((option.entity_text_zh or "").strip(), (option.entity_text_en or "").strip())
        grouped.setdefault(key, []).append(option)

    merged: list[AttributeCandidate] = []
    for group in grouped.values():
        best = max(group, key=lambda item: float(item.normalized_score or 0.0))
        total_normalized = round(sum(float(item.normalized_score or 0.0) for item in group), 4)
        merged.append(
            best.model_copy(
                update={
                    "score": total_normalized,
                    "normalized_score": total_normalized,
                    "display_score": total_normalized,
                }
            )
        )

    merged.sort(key=lambda item: float(item.normalized_score or 0.0), reverse=True)
    ranked = []
    for idx, option in enumerate(merged[:topk], start=1):
        ranked.append(option.model_copy(update={"rank": idx}))
    return ranked


def _predict_relation(entity_id: str, relation_id: str, *, topk: int) -> tuple[list[AttributeCandidate], int]:
    predictor = get_predictor()
    candidate_ids = relation_tail_candidates().get(relation_id, [])
    candidate_count = len(candidate_ids)
    if not candidate_ids:
        return [], 0

    import torch

    head = predictor.parse_entity(entity_id)
    rel = predictor.parse_relation(relation_id)
    tail_ids = [predictor.parse_entity(candidate) for candidate in candidate_ids]
    device = predictor.device

    h = torch.full((len(tail_ids),), head, dtype=torch.long, device=device)
    r = torch.full((len(tail_ids),), rel, dtype=torch.long, device=device)
    t = torch.tensor(tail_ids, dtype=torch.long, device=device)
    triples = torch.stack([h, r, t], dim=1)

    start = perf_counter()
    with torch.inference_mode():
        raw_scores_tensor = predictor.model.score(triples).detach().cpu()
    elapsed_ms = round((perf_counter() - start) * 1000, 3)

    raw_scores = [float(value) for value in raw_scores_tensor.tolist()]
    normalized_scores = score_to_normalized(raw_scores)
    ranked_indices = sorted(range(len(candidate_ids)), key=lambda idx: normalized_scores[idx], reverse=True)

    logger.info(
        "Attribute relation scored: entity=%s relation=%s candidate_count=%s latency_ms=%s",
        entity_id,
        relation_id,
        candidate_count,
        elapsed_ms,
    )

    options = [
        _candidate(
            candidate_ids[idx],
            raw_score=raw_scores[idx],
            normalized_score=normalized_scores[idx],
            rank=rank,
        )
        for rank, idx in enumerate(ranked_indices[:topk], start=1)
    ]
    return _dedupe_options(options, topk=topk), candidate_count


def _fallback_attribute_relations(entity_id: str, topk: int) -> list[AttributeRow]:
    observed = triples_by_pair()
    rows: list[AttributeRow] = []

    for relation_id in sorted({relation for head, relation in observed if head == entity_id}):
        existing = observed.get((entity_id, relation_id), [])
        if not existing:
            continue

        options = [
            _candidate(entity, raw_score=None, normalized_score=None, rank=rank)
            for rank, entity in enumerate(existing[:topk], start=1)
        ]
        if not options:
            continue

        rows.append(
            AttributeRow(
                relation_id=relation_id,
                relation_name=relation_text(relation_id),
                relation_name_en=relation_text_en(relation_id),
                source="existing",
                selected_option_index=0,
                selected_value=options[0],
                options=options,
                candidate_count=len(existing),
                warning="Model artifacts are not ready; showing observed attributes only.",
            )
        )

    return rows


def get_attribute_completion(entity_id: str, topk: int) -> AttributeCompletionResponse:
    start = perf_counter()
    cfg = get_runtime_config()
    _ensure_entity(entity_id)

    status = predictor_status()
    if not status["model_ready"]:
        text_zh = entity_text(entity_id)
        text_en = entity_text_en(entity_id)
        image_path = image_path_for_entity(entity_id)
        rows = _fallback_attribute_relations(entity_id, topk)
        latency_ms = round((perf_counter() - start) * 1000, 3)
        return AttributeCompletionResponse(
            model=f"{cfg['model_name']} (metadata only)",
            device=cfg["device"],
            inputs={
                "entity": entity_id,
                "entity_text": text_zh,
                "entity_text_zh": text_zh,
                "entity_text_en": text_en,
                "topk": topk,
                "attribute_relations": [row.relation_id for row in rows],
                "model_status": status,
            },
            results={
                "entity_info": {
                    "entity": entity_id,
                    "entity_text": text_zh,
                    "entity_text_zh": text_zh,
                    "entity_text_en": text_en,
                    "has_image": image_path is not None,
                    "image_path": image_path,
                },
                "attribute_rows": [row.model_dump() for row in rows],
            },
            latency_ms=latency_ms,
        )

    predictor = get_predictor()
    relations = _attribute_relations(predictor)
    observed = triples_by_pair()

    logger.info(
        "Attribute completion start: entity=%s run_dir=%s attribute_relations=%s",
        entity_id,
        status["run_dir"],
        len(relations),
    )

    rows: list[AttributeRow] = []
    for relation_id in relations:
        relation_name = relation_text(relation_id)
        relation_name_en = relation_text_en(relation_id)
        existing = observed.get((entity_id, relation_id), [])

        warning = None
        candidate_count = relation_tail_count(relation_id)

        if existing:
            options = [
                _candidate(entity, raw_score=None, normalized_score=None, rank=rank)
                for rank, entity in enumerate(existing[:topk], start=1)
            ]
            source = "existing"
        else:
            options, candidate_count = _predict_relation(entity_id, relation_id, topk=topk)
            source = "predicted"
            if candidate_count == 0:
                warning = "No candidate set available for this relation."
            elif not options:
                warning = "Model returned no ranked candidate for this relation."

        if not options:
            if warning is None:
                warning = "No available value for this relation."
            placeholder = _candidate(entity="", raw_score=None, normalized_score=None, rank=None)
            rows.append(
                AttributeRow(
                    relation_id=relation_id,
                    relation_name=relation_name,
                    relation_name_en=relation_name_en,
                    source="predicted",
                    selected_option_index=0,
                    selected_value=placeholder,
                    options=[],
                    candidate_count=candidate_count,
                    warning=warning,
                )
            )
            continue

        rows.append(
            AttributeRow(
                relation_id=relation_id,
                relation_name=relation_name,
                relation_name_en=relation_name_en,
                source=source,
                selected_option_index=0,
                selected_value=options[0],
                options=options,
                candidate_count=candidate_count,
                warning=warning,
            )
        )

    text_zh = entity_text(entity_id)
    text_en = entity_text_en(entity_id)
    image_path = image_path_for_entity(entity_id)
    latency_ms = round((perf_counter() - start) * 1000, 3)

    return AttributeCompletionResponse(
        model=cfg["model_name"],
        device=cfg["device"],
        inputs={
            "entity": entity_id,
            "entity_text": text_zh,
            "entity_text_zh": text_zh,
            "entity_text_en": text_en,
            "topk": topk,
            "attribute_relations": relations,
        },
        results={
            "entity_info": {
                "entity": entity_id,
                "entity_text": text_zh,
                "entity_text_zh": text_zh,
                "entity_text_en": text_en,
                "has_image": image_path is not None,
                "image_path": image_path,
            },
            "attribute_rows": [row.model_dump() for row in rows],
        },
        latency_ms=latency_ms,
    )
