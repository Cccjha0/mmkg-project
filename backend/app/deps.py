from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def _seed_name_from_run_dir(path: Path) -> str | None:
    marker = "_seed"
    if marker not in path.name:
        return None
    seed = path.name.rsplit(marker, 1)[-1]
    return f"seed{seed}" if seed.isdigit() else None


def _valid_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config_merged.json").is_file() and (path / "best.ckpt").is_file()


def _best_seed_by_summary(model_code: str) -> str | None:
    summary_path = repo_root() / "ml" / "artifacts" / "plots" / model_code / "best_summary.csv"
    if not summary_path.is_file():
        return None

    best_seed: str | None = None
    best_mrr = float("-inf")
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            seed = (row.get("seed") or "").strip()
            if not seed.startswith("seed"):
                continue
            try:
                mrr = float(row.get("best_mrr") or "")
            except ValueError:
                continue
            if mrr > best_mrr:
                best_seed = seed
                best_mrr = mrr

    return best_seed


def _default_production_run_dir(model_code: str) -> str:
    model_dir = repo_root() / "ml" / "artifacts" / "production_models" / model_code
    valid_runs = [path for path in model_dir.iterdir() if _valid_run_dir(path)] if model_dir.is_dir() else []
    if not valid_runs:
        return f"ml/artifacts/production_models/{model_code}"

    best_seed = _best_seed_by_summary(model_code)
    if best_seed is not None:
        for run_dir in valid_runs:
            if _seed_name_from_run_dir(run_dir) == best_seed:
                return str(run_dir.relative_to(repo_root()))

    newest_run = max(valid_runs, key=lambda path: path.stat().st_mtime)
    return str(newest_run.relative_to(repo_root()))


@lru_cache(maxsize=1)
def get_runtime_config() -> dict[str, Any]:
    model_code = os.getenv("MMKG_MODEL_CODE", "gate+residual")
    return {
        "run_dir": os.getenv(
            "MMKG_RUN_DIR",
            _default_production_run_dir(model_code),
        ),
        "device": os.getenv("MMKG_DEVICE", "cpu"),
        "dataset": os.getenv("MMKG_DATASET", "OpenBG-IMG"),
        "dataset_name": "openbg_img",
        "model_name": os.getenv("MMKG_MODEL_NAME", "Residual+Gate"),
        "model_code": model_code,
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root() / path


def backend_data_path(*parts: str) -> Path:
    return repo_root() / "backend" / "data" / Path(*parts)


def openbg_img_raw_path(*parts: str) -> Path:
    return repo_root() / "data" / "datasets" / "openbg_img" / "raw" / Path(*parts)


def openbg_img_cache_path(*parts: str) -> Path:
    return repo_root() / "data" / "cache" / "openbg_img" / Path(*parts)


def artifacts_path(*parts: str) -> Path:
    return repo_root() / "ml" / "artifacts" / Path(*parts)


def runtime_run_dir_path() -> Path:
    return resolve_project_path(get_runtime_config()["run_dir"])


def runtime_config_path() -> Path:
    return runtime_run_dir_path() / "config_merged.json"


def runtime_checkpoint_path() -> Path:
    return runtime_run_dir_path() / "best.ckpt"


def read_json_file(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
