from __future__ import annotations

from fastapi import APIRouter, Path, Query

from app.schemas.attribute_completion import AttributeCompletionResponse
from app.services.attribute_service import get_attribute_completion

router = APIRouter(prefix="/api/entities", tags=["attribute-completion"])


@router.get(
    "/{entity_id}/attribute-completion",
    response_model=AttributeCompletionResponse,
    summary="Attribute completion",
)
def api_get_attribute_completion(
    entity_id: str = Path(..., description="Entity id, for example ent_007314"),
    topk: int = Query(default=5, ge=1, le=20, description="Maximum predicted candidates per relation"),
) -> AttributeCompletionResponse:
    return get_attribute_completion(entity_id=entity_id, topk=topk)
