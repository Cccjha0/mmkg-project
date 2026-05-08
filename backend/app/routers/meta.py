from __future__ import annotations

from fastapi import APIRouter

from app.schemas.meta import ModelMetaResponse, RuntimeResponse
from app.services.meta_service import get_model_meta, get_runtime_info

router = APIRouter(prefix="/api/meta", tags=["meta"])


@router.get("/runtime", response_model=RuntimeResponse, summary="Runtime information")
def api_get_runtime_info() -> RuntimeResponse:
    return get_runtime_info()


@router.get("/model", response_model=ModelMetaResponse, summary="Model metadata")
def api_get_model_metadata() -> ModelMetaResponse:
    return get_model_meta()
