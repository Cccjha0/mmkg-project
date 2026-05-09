from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.services import performance_service


def _write_summary(root: Path, model_key: str, *, mrr: str, hits1: str, hits3: str, hits10: str) -> None:
    path = root / "plots" / model_key / "best_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "seed,best_epoch,best_mrr,hits@1_at_best,hits@3_at_best,hits@10_at_best",
                f"seed1,3,{mrr},{hits1},{hits3},{hits10}",
                f"mean+/-std,3,{mrr} +/- 0.01,{hits1} +/- 0.01,{hits3} +/- 0.01,{hits10} +/- 0.01",
            ]
        ),
        encoding="utf-8",
    )


def test_performance_overview_and_comparison_parse_mock_summaries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(performance_service, "artifacts_path", lambda *parts: tmp_path.joinpath(*parts))
    _write_summary(tmp_path, "gate+residual", mrr="0.50", hits1="0.35", hits3="0.58", hits10="0.78")
    _write_summary(tmp_path, "gate-only", mrr="0.36", hits1="0.22", hits3="0.43", hits10="0.65")

    overview = performance_service.get_performance_overview()
    comparison = performance_service.get_model_comparison(dataset="OpenBG-IMG")

    assert overview.best_accuracy_model == "Residual+Gate"
    assert overview.best_accuracy == 0.78
    assert overview.num_models == 2
    assert comparison.metric == "accuracy"
    assert [row.model for row in comparison.rows] == ["Gate Only", "Residual+Gate"]
    assert comparison.rows[-1].mrr == 0.5


def test_accuracy_curves_load_from_plot_input_when_metrics_are_absent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(performance_service, "artifacts_path", lambda *parts: tmp_path.joinpath(*parts))
    seed_path = tmp_path / "plot_input" / "gate+residual" / "seed1.csv"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(
        "epoch,hits@10\n1,0.4\n2,0.6\n",
        encoding="utf-8",
    )

    response = performance_service.get_accuracy_curves(dataset="OpenBG-IMG")

    assert response.dataset == "OpenBG-IMG"
    assert response.epochs == [1, 2]
    assert response.series[0].model == "Residual+Gate"
    assert response.series[0].values == [0.4, 0.6]


def test_model_comparison_rejects_supported_dataset_with_no_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(performance_service, "artifacts_path", lambda *parts: tmp_path.joinpath(*parts))

    with pytest.raises(HTTPException) as exc:
        performance_service.get_model_comparison(dataset="OpenBG-IMG")

    assert exc.value.status_code == 400
    assert exc.value.detail["error"]["code"] == "INVALID_ARGUMENT"
