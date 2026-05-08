from __future__ import annotations

from fastapi import APIRouter

from app.schemas.common import HealthResponse
from app.services.meta_service import get_health_info

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
def get_health() -> HealthResponse:
    return get_health_info()
