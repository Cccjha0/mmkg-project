import json
from pathlib import Path
from typing import Any

import torch

from ml.inference.predictor import Predictor
from ml.training.src.models.build_model import build_model


def resolve_device(requested: str | None = None) -> str:
    requested = (requested or "cpu").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    return "cpu"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_project_path(raw: str | None) -> str | None:
    if not raw:
        return raw

    path = Path(raw)
    if path.is_absolute():
        return str(path)

    root = _repo_root()
    direct = root / path
    if direct.exists():
        return str(direct)

    raw_norm = raw.replace("\\", "/")
    legacy_prefixes = {
        "cache/": "data/cache/",
        "datasets/": "data/datasets/",
    }
    for old_prefix, new_prefix in legacy_prefixes.items():
        if raw_norm.startswith(old_prefix):
            rewritten = root / Path(new_prefix + raw_norm[len(old_prefix):])
            return str(rewritten)

    return str(direct)


def _normalize_cfg_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset = cfg.get("dataset", {})
    for key in ("train", "dev", "test", "cache_dir"):
        if key in dataset:
            dataset[key] = _normalize_project_path(dataset.get(key))

    model = cfg.get("model", {})
    for key in ("bert_cache_path",):
        if key in model:
            model[key] = _normalize_project_path(model.get(key))

    config_paths = cfg.get("_config_paths", {})
    for key in ("common", "exp"):
        if key in config_paths:
            config_paths[key] = _normalize_project_path(config_paths.get(key))

    return cfg


def _resolve_run_artifacts(
    run_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[Path, Path]:
    cfg_path = Path(config_path) if config_path else None
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None

    if run_dir is not None:
        run_path = Path(run_dir)
        if cfg_path is None:
            cfg_path = run_path / "config_merged.json"
        if ckpt_path is None:
            ckpt_path = run_path / "best.ckpt"

    if cfg_path is None or ckpt_path is None:
        raise ValueError("Provide run_dir or both config_path and checkpoint_path.")

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    return cfg_path, ckpt_path


def load_runtime(
    run_dir: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    cfg_path, ckpt_path = _resolve_run_artifacts(
        run_dir=run_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    cfg = _normalize_cfg_paths(_load_json(cfg_path))
    resolved_device = resolve_device(device)
    cfg.setdefault("system", {})
    cfg["system"]["device"] = resolved_device

    model, num_entities = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(resolved_device)
    model.eval()

    return {
        "model": model,
        "cfg": cfg,
        "device": resolved_device,
        "num_entities": num_entities,
        "config_path": cfg_path,
        "checkpoint_path": ckpt_path,
        "run_dir": ckpt_path.parent,
    }


def load_predictor(
    run_dir: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> Predictor:
    runtime = load_runtime(
        run_dir=run_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return Predictor(
        model=runtime["model"],
        cfg=runtime["cfg"],
        device=runtime["device"],
        num_entities=runtime["num_entities"],
        run_dir=runtime["run_dir"],
    )
