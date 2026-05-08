from __future__ import annotations

from fastapi import APIRouter, Query

from app.schemas.performance import (
    AccuracyCurvesResponse,
    AccuracyDatasetResponse,
    DatasetQuery,
    ModelComparisonResponse,
    PerformanceOverviewResponse,
)
from app.services.performance_service import (
    get_accuracy_curves,
    get_model_comparison,
    get_performance_overview,
)

router = APIRouter(prefix="/api/performance", tags=["performance"])


@router.get("/overview", response_model=PerformanceOverviewResponse, summary="Performance overview")
def api_get_performance_overview() -> PerformanceOverviewResponse:
    return get_performance_overview()


@router.get(
    "/accuracy-curves",
    response_model=AccuracyCurvesResponse | AccuracyDatasetResponse,
    summary="Accuracy curves",
)
def api_get_accuracy_curves(
    dataset: DatasetQuery = Query(default="all", description="all, OpenBG-500, or OpenBG-IMG"),
) -> AccuracyCurvesResponse | AccuracyDatasetResponse:
    return get_accuracy_curves(dataset=dataset)


@router.get("/model-comparison", response_model=ModelComparisonResponse, summary="Model comparison")
def api_get_model_comparison(
    dataset: DatasetQuery = Query(default="all", description="all, OpenBG-500, or OpenBG-IMG"),
) -> ModelComparisonResponse:
    return get_model_comparison(dataset=dataset)
