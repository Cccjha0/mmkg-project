from __future__ import annotations

import logging
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

_MODEL_LOAD_ERROR: str | None = None


def _resolve_run_dir() -> Path:
    from app.deps import runtime_run_dir_path

    return runtime_run_dir_path()


def _config_path() -> Path:
    from app.deps import runtime_config_path

    return runtime_config_path()


def _checkpoint_path() -> Path:
    from app.deps import runtime_checkpoint_path

    return runtime_checkpoint_path()


def _load_predictor_impl():
    from ml.inference.runtime import load_predictor

    run_dir = _resolve_run_dir()
    cfg_path = _config_path()
    ckpt_path = _checkpoint_path()

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    logger.info("Loading predictor: run_dir=%s checkpoint=%s", run_dir, ckpt_path)
    return load_predictor(run_dir=run_dir, device="cpu")


@lru_cache(maxsize=1)
def get_predictor():
    global _MODEL_LOAD_ERROR
    try:
        predictor = _load_predictor_impl()
        _MODEL_LOAD_ERROR = None
        return predictor
    except Exception as exc:
        _MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed to load predictor")
        raise


def predictor_status() -> dict[str, Any]:
    run_dir = _resolve_run_dir()
    cfg_path = _config_path()
    ckpt_path = _checkpoint_path()
    ready = run_dir.is_dir() and cfg_path.is_file() and ckpt_path.is_file()
    return {
        "run_dir": str(run_dir),
        "config_path": str(cfg_path),
        "checkpoint_path": str(ckpt_path),
        "model_ready": ready,
        "load_error": _MODEL_LOAD_ERROR,
    }


def score_to_display(score: float) -> float:
    return round(1.0 / (1.0 + math.exp(-float(score))), 4)


def score_to_normalized(scores: list[float]) -> list[float]:
    if not scores:
        return []
    exps = [math.exp(float(score) - max(scores)) for score in scores]
    denom = sum(exps)
    if denom <= 0:
        return [0.0 for _ in scores]
    return [round(value / denom, 4) for value in exps]
