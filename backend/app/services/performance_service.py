from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from app.deps import artifacts_path
from app.schemas.performance import (
    AccuracyCurvesResponse,
    AccuracyDatasetResponse,
    AccuracySeries,
    DatasetGroup,
    DatasetQuery,
    ModelComparisonResponse,
    ModelComparisonRow,
    PerformanceOverviewResponse,
)

MODEL_LABELS = {
    "text-only": "Text-ComplEx",
    "text-rgcn": "Text-RGCN",
    "openbg_img_early": "Early Fusion",
    "gate-only": "Gate Only",
    "gate+residual": "Residual+Gate",
}

MODEL_DATASETS = {
    "text-only": "OpenBG-500",
    "text-rgcn": "OpenBG-500",
    "openbg_img_early": "OpenBG-IMG",
    "gate-only": "OpenBG-IMG",
    "gate+residual": "OpenBG-IMG",
}

ALLOWED_DATASETS = ["all", "OpenBG-500", "OpenBG-IMG"]


def _bad_dataset(dataset: str) -> None:
    raise HTTPException(
        status_code=400,
        detail={
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": f"Unsupported dataset: {dataset}",
                "details": {"allowed": ALLOWED_DATASETS},
            }
        },
    )


def _parse_mean(value: str) -> float:
    return float(value.split("+/-", 1)[0].strip())


def _best_epoch_for_model(model_dir: str) -> int:
    path = artifacts_path("plots", model_dir, "best_summary.csv")
    if not path.is_file():
        return 1

    epochs: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            seed = row.get("seed", "")
            if seed.startswith("seed") and row.get("best_epoch"):
                epochs.append(float(row["best_epoch"]))

    return int(round(sum(epochs) / len(epochs))) if epochs else 1


def _best_summary_rows() -> list[ModelComparisonRow]:
    rows: list[ModelComparisonRow] = []
    for model_dir, label in MODEL_LABELS.items():
        path = artifacts_path("plots", model_dir, "best_summary.csv")
        if not path.is_file():
            continue

        with path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("seed") != "mean+/-std":
                    continue

                rows.append(
                    ModelComparisonRow(
                        model=label,
                        dataset=MODEL_DATASETS[model_dir],
                        accuracy=_parse_mean(row["hits@10_at_best"]),
                        mrr=_parse_mean(row["best_mrr"]),
                        hits1=_parse_mean(row["hits@1_at_best"]),
                        hits3=_parse_mean(row["hits@3_at_best"]),
                        hits10=_parse_mean(row["hits@10_at_best"]),
                        best_epoch=_best_epoch_for_model(model_dir),
                    )
                )
    return rows


def _metrics_csv_for_model(model_dir: str) -> Path | None:
    candidates = sorted(artifacts_path("outputs", model_dir).glob("*/metrics.csv"))
    return candidates[0] if candidates else None


def _load_accuracy_dataset(dataset: str) -> AccuracyDatasetResponse:
    epochs: list[int] = []
    series: list[AccuracySeries] = []

    for model_key, label in MODEL_LABELS.items():
        if MODEL_DATASETS[model_key] != dataset:
            continue

        metrics_path = _metrics_csv_for_model(model_key)
        if metrics_path is None:
            continue

        with metrics_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))

        model_epochs = [int(float(row["epoch"])) for row in rows]
        values = [float(row["hits@10"]) for row in rows]

        if not epochs:
            epochs = model_epochs

        limit = min(len(epochs), len(values))
        if limit == 0:
            continue

        if len(model_epochs) < len(epochs):
            epochs = model_epochs[:limit]
            series = [
                AccuracySeries(model=item.model, values=item.values[:limit])
                for item in series
            ]

        series.append(AccuracySeries(model=label, values=values[: len(epochs)]))

    return AccuracyDatasetResponse(
        metric="accuracy",
        dataset=dataset,
        epochs=epochs,
        series=series,
    )


def get_performance_overview() -> PerformanceOverviewResponse:
    rows = _best_summary_rows()
    if not rows:
        return PerformanceOverviewResponse(
            dataset_groups=[],
            best_accuracy_model="",
            best_accuracy=0.0,
            num_models=0,
            last_updated="",
        )

    best = max(rows, key=lambda row: row.accuracy)
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row.dataset, []).append(row.model)

    latest = max(
        (
            artifacts_path("plots", key, "best_summary.csv").stat().st_mtime
            for key in MODEL_LABELS
            if artifacts_path("plots", key, "best_summary.csv").is_file()
        ),
        default=0,
    )

    return PerformanceOverviewResponse(
        dataset_groups=[
            DatasetGroup(dataset=dataset, models=models)
            for dataset, models in groups.items()
        ],
        best_accuracy_model=best.model,
        best_accuracy=best.accuracy,
        num_models=len(rows),
        last_updated=datetime.fromtimestamp(latest).strftime("%Y-%m-%d") if latest else "",
    )


def get_accuracy_curves(dataset: DatasetQuery = "all") -> AccuracyCurvesResponse | AccuracyDatasetResponse:
    if dataset == "all":
        datasets = {
            name: _load_accuracy_dataset(name)
            for name in ALLOWED_DATASETS
            if name != "all"
        }
        return AccuracyCurvesResponse(metric="accuracy", datasets=datasets)

    if dataset not in ALLOWED_DATASETS:
        _bad_dataset(dataset)

    response = _load_accuracy_dataset(dataset)
    if not response.series:
        _bad_dataset(dataset)
    return response


def get_model_comparison(dataset: DatasetQuery = "all") -> ModelComparisonResponse:
    rows = _best_summary_rows()
    if dataset != "all":
        if dataset not in ALLOWED_DATASETS:
            _bad_dataset(dataset)
        rows = [row for row in rows if row.dataset == dataset]
        if not rows:
            _bad_dataset(dataset)
    return ModelComparisonResponse(metric="accuracy", rows=rows)
