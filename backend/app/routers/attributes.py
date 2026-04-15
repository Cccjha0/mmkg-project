from __future__ import annotations

from fastapi import APIRouter, Path, Query

from app.schemas.attribute_completion import AttributeCompletionResponse
from app.services.attribute_service import get_attribute_completion

router = APIRouter(prefix="/api/entities", tags=["attribute-completion"])


@router.get(
    "/{entity_id}/attribute-completion",
    response_model=AttributeCompletionResponse,
    summary="获取属性补全表格数据",
)
def api_get_attribute_completion(
    entity_id: str = Path(..., description="实体 ID，例如 ent_007314"),
    topk: int = Query(default=5, ge=1, le=20, description="predicted 属性候选数量"),
) -> AttributeCompletionResponse:
    return get_attribute_completion(entity_id=entity_id, topk=topk)
