from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def get_runtime_config() -> dict[str, Any]:
    return {
        "run_dir": os.getenv(
            "MMKG_RUN_DIR",
            "ml/artifacts/outputs/gate+residual/20260314_212911_seed1",
        ),
        "device": os.getenv("MMKG_DEVICE", "cpu"),
        "dataset": os.getenv("MMKG_DATASET", "OpenBG-IMG"),
        "dataset_name": "openbg_img",
        "model_name": os.getenv("MMKG_MODEL_NAME", "Residual+Gate"),
        "model_code": os.getenv("MMKG_MODEL_CODE", "gate+residual"),
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
