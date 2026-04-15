from __future__ import annotations

from fastapi import APIRouter, Query

from app.schemas.performance import (
    AccuracyTrendResponse,
    DatasetName,
    ModelComparisonResponse,
    PerformanceSummaryResponse,
)
from app.services.performance_service import (
    get_accuracy_trends,
    get_model_comparison,
    get_performance_summary,
)

router = APIRouter(prefix="/api/performance", tags=["performance"])


@router.get("/summary", response_model=PerformanceSummaryResponse, summary="获取页面摘要信息")
def api_get_performance_summary(
    dataset: DatasetName = Query(..., description="数据集名"),
) -> PerformanceSummaryResponse:
    return get_performance_summary(dataset=dataset)


@router.get("/model-comparison", response_model=ModelComparisonResponse, summary="获取模型对比柱状图数据")
def api_get_model_comparison(
    dataset: DatasetName = Query(..., description="数据集名"),
) -> ModelComparisonResponse:
    return get_model_comparison(dataset=dataset)


@router.get("/accuracy-trends", response_model=AccuracyTrendResponse, summary="获取多模型 Accuracy 折线图数据")
def api_get_accuracy_trends(
    dataset: DatasetName = Query(..., description="数据集名"),
) -> AccuracyTrendResponse:
    return get_accuracy_trends(dataset=dataset)
