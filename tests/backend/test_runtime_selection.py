from __future__ import annotations

from pathlib import Path

from app import deps


def _make_run(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config_merged.json").write_text("{}", encoding="utf-8")
    (path / "best.ckpt").write_bytes(b"checkpoint")


def test_default_run_dir_uses_best_summary_seed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(deps, "repo_root", lambda: tmp_path)
    model_dir = tmp_path / "ml" / "artifacts" / "production_models" / "gate+residual"
    _make_run(model_dir / "20260314_111111_seed1")
    _make_run(model_dir / "20260314_222222_seed2")
    summary_dir = tmp_path / "ml" / "artifacts" / "plots" / "gate+residual"
    summary_dir.mkdir(parents=True)
    (summary_dir / "best_summary.csv").write_text(
        "seed,best_mrr\nseed1,0.4\nseed2,0.7\n",
        encoding="utf-8",
    )

    selected = deps._default_production_run_dir("gate+residual")

    assert Path(selected) == Path("ml/artifacts/production_models/gate+residual/20260314_222222_seed2")


def test_default_run_dir_falls_back_to_outputs_for_gate_residual(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(deps, "repo_root", lambda: tmp_path)
    fallback_dir = tmp_path / "ml" / "artifacts" / "outputs" / "openbg_img_gated_vec_res_rel"
    _make_run(fallback_dir / "20260315_111111_seed1")

    selected = deps._default_production_run_dir("gate+residual")

    assert Path(selected) == Path("ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260315_111111_seed1")


def test_runtime_config_respects_manual_run_dir(monkeypatch) -> None:
    deps.get_runtime_config.cache_clear()
    monkeypatch.setenv("MMKG_RUN_DIR", "custom/run")
    monkeypatch.setenv("MMKG_MODEL_CODE", "gate+residual")

    try:
        cfg = deps.get_runtime_config()
    finally:
        deps.get_runtime_config.cache_clear()

    assert cfg["run_dir"] == "custom/run"
    assert cfg["model_code"] == "gate+residual"
