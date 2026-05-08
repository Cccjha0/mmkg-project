from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DatasetQuery = Literal["all", "OpenBG-500", "OpenBG-IMG"]


class DatasetGroup(BaseModel):
    dataset: str
    models: list[str]


class PerformanceOverviewResponse(BaseModel):
    dataset_groups: list[DatasetGroup]
    best_accuracy_model: str
    best_accuracy: float = Field(..., ge=0.0, le=1.0)
    num_models: int = Field(..., ge=0)
    last_updated: str


class AccuracySeries(BaseModel):
    model: str
    values: list[float]


class AccuracyDatasetResponse(BaseModel):
    metric: Literal["accuracy"] = "accuracy"
    dataset: str
    epochs: list[int]
    series: list[AccuracySeries]


class AccuracyCurvesResponse(BaseModel):
    metric: Literal["accuracy"] = "accuracy"
    datasets: dict[str, AccuracyDatasetResponse]


class ModelComparisonRow(BaseModel):
    model: str
    dataset: str
    accuracy: float = Field(..., ge=0.0, le=1.0)
    mrr: float = Field(..., ge=0.0, le=1.0)
    hits1: float = Field(..., ge=0.0, le=1.0)
    hits3: float = Field(..., ge=0.0, le=1.0)
    hits10: float = Field(..., ge=0.0, le=1.0)
    best_epoch: int = Field(..., ge=1)


class ModelComparisonResponse(BaseModel):
    metric: Literal["accuracy"] = "accuracy"
    rows: list[ModelComparisonRow]
