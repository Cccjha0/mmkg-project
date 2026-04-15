from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DatasetName = Literal["openbg500", "openbg_img"]
MetricName = Literal["accuracy"]


class PerformanceFilterQuery(BaseModel):
    dataset: DatasetName = Field(..., description="数据集名")
    metric: MetricName = Field(default="accuracy", description="当前图表指标，第一版固定为 accuracy")


class PerformanceSummaryItem(BaseModel):
    label: str = Field(..., description="摘要项名称")
    value: str = Field(..., description="摘要项数值或文本")


class PerformanceSummaryResponse(BaseModel):
    dataset: DatasetName
    metric: MetricName = "accuracy"
    summary_cards: list[PerformanceSummaryItem] = Field(default_factory=list, description="顶部摘要卡片")
    summary_rows: list[PerformanceSummaryItem] = Field(default_factory=list, description="下方摘要信息")


class ModelMetricPoint(BaseModel):
    model_key: str = Field(..., description="模型唯一标识")
    model_name: str = Field(..., description="模型展示名称")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="最终 accuracy 指标")


class ModelComparisonResponse(BaseModel):
    dataset: DatasetName
    metric: MetricName = "accuracy"
    items: list[ModelMetricPoint] = Field(default_factory=list, description="模型对比柱状图数据")


class AccuracyTrendPoint(BaseModel):
    epoch: int = Field(..., ge=1, description="训练轮次")
    value: float = Field(..., ge=0.0, le=1.0, description="accuracy 值")


class AccuracyTrendSeries(BaseModel):
    model_key: str = Field(..., description="模型唯一标识")
    model_name: str = Field(..., description="模型展示名称")
    points: list[AccuracyTrendPoint] = Field(default_factory=list, description="折线图点列")


class AccuracyTrendResponse(BaseModel):
    dataset: DatasetName
    metric: MetricName = "accuracy"
    series: list[AccuracyTrendSeries] = Field(default_factory=list, description="多模型 accuracy 折线图")
