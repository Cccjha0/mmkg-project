from __future__ import annotations

from fastapi import APIRouter

from app.schemas.meta import ModelMetaResponse
from app.services.meta_service import get_model_meta

router = APIRouter(prefix="/api/meta", tags=["meta"])


@router.get("/model", response_model=ModelMetaResponse, summary="获取当前模型元信息")
def get_model_metadata() -> ModelMetaResponse:
    return get_model_meta()
