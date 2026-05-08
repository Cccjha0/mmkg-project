from __future__ import annotations

from fastapi import APIRouter, Path, Query

from app.schemas.entity import EmbeddingSpace, EntityInfoResponse, SimilarEntitiesResponse
from app.services.entity_service import get_entity_info, get_similar_entities

router = APIRouter(prefix="/api/entities", tags=["entities"])


@router.get("/{entity_id}", response_model=EntityInfoResponse, summary="Entity detail")
def api_get_entity_info(
    entity_id: str = Path(..., description="Entity id, for example ent_007314"),
) -> EntityInfoResponse:
    return get_entity_info(entity_id=entity_id)


@router.get("/{entity_id}/similar", response_model=SimilarEntitiesResponse, summary="Similar entities")
def api_get_similar_entities(
    entity_id: str = Path(..., description="Entity id, for example ent_007314"),
    space: EmbeddingSpace = Query(default="fused", description="text, image, fused, or entity_repr"),
    topk: int = Query(default=10, ge=1, le=20),
) -> SimilarEntitiesResponse:
    return get_similar_entities(entity_id=entity_id, space=space, topk=topk)
