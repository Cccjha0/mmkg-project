from __future__ import annotations

from app.schemas.performance import (
    AccuracyTrendPoint,
    AccuracyTrendResponse,
    AccuracyTrendSeries,
    DatasetName,
    ModelComparisonResponse,
    ModelMetricPoint,
    PerformanceSummaryItem,
    PerformanceSummaryResponse,
)


def get_performance_summary(dataset: DatasetName) -> PerformanceSummaryResponse:
    """
    TODO:
    1. 从标准化 summary json/csv 读取真实实验结果
    2. 根据 dataset 过滤
    3. 生成顶部卡片和下方摘要信息
    """
    return PerformanceSummaryResponse(
        dataset=dataset,
        summary_cards=[
            PerformanceSummaryItem(label="最佳 Accuracy", value="0.000"),
            PerformanceSummaryItem(label="参与模型数", value="0"),
            PerformanceSummaryItem(label="数据集", value=dataset),
        ],
        summary_rows=[
            PerformanceSummaryItem(label="当前状态", value="待接入真实实验结果"),
        ],
    )


def get_model_comparison(dataset: DatasetName) -> ModelComparisonResponse:
    """
    TODO:
    读取每个模型的最终 accuracy，用于柱状图。
    """
    return ModelComparisonResponse(
        dataset=dataset,
        items=[
            ModelMetricPoint(model_key="placeholder", model_name="Placeholder", accuracy=0.0),
        ],
    )


def get_accuracy_trends(dataset: DatasetName) -> AccuracyTrendResponse:
    """
    TODO:
    读取多模型逐 epoch accuracy 曲线。
    """
    return AccuracyTrendResponse(
        dataset=dataset,
        series=[
            AccuracyTrendSeries(
                model_key="placeholder",
                model_name="Placeholder",
                points=[
                    AccuracyTrendPoint(epoch=1, value=0.0),
                ],
            )
        ],
    )
