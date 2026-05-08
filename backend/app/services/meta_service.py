from __future__ import annotations

import json

from app.deps import get_runtime_config, runtime_config_path
from app.schemas.common import HealthResponse
from app.schemas.meta import ModelMetaResponse, RuntimeResponse
from app.services.inference_service import predictor_status


def _attribute_relations_from_config() -> list[str]:
    cfg_path = runtime_config_path()
    if not cfg_path.is_file():
        return []
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return cfg.get("inference", {}).get("attribute_relations", []) or []


def get_runtime_info() -> RuntimeResponse:
    cfg = get_runtime_config()
    status = predictor_status()
    warnings: list[str] = []
    if status.get("load_error"):
        warnings.append(status["load_error"])

    return RuntimeResponse(
        model_name=cfg["model_name"],
        model_code=cfg["model_code"],
        dataset=cfg["dataset"],
        run_dir=status["run_dir"],
        config_path=status["config_path"],
        checkpoint_path=status["checkpoint_path"],
        device=cfg["device"],
        attribute_relations=_attribute_relations_from_config(),
        model_ready=bool(status["model_ready"]),
        warnings=warnings,
    )


def get_model_meta() -> ModelMetaResponse:
    runtime = get_runtime_info()
    return ModelMetaResponse(**runtime.model_dump())


def get_health_info() -> HealthResponse:
    cfg = get_runtime_config()
    runtime = get_runtime_info()
    return HealthResponse(
        model_loaded=runtime.model_ready,
        dataset=cfg["dataset"],
        model=cfg["model_name"],
        run_dir=runtime.run_dir,
        warnings=runtime.warnings,
    )
