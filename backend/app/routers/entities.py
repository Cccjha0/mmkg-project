from __future__ import annotations

from fastapi import APIRouter, Path

from app.schemas.entity import EntityInfoResponse
from app.services.entity_service import get_entity_info

router = APIRouter(prefix="/api/entities", tags=["entities"])


@router.get("/{entity_id}", response_model=EntityInfoResponse, summary="获取实体基础信息")
def api_get_entity_info(
    entity_id: str = Path(..., description="实体 ID，例如 ent_007314"),
) -> EntityInfoResponse:
    return get_entity_info(entity_id=entity_id)
