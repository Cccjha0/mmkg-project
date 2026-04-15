from __future__ import annotations

from app.deps import get_runtime_config
from app.schemas.meta import ModelMetaResponse


def get_model_meta() -> ModelMetaResponse:
    cfg = get_runtime_config()
    return ModelMetaResponse(
        model_name=cfg["model_name"],
        dataset_name=cfg["dataset_name"],
        device=cfg["device"],
        run_dir=cfg["run_dir"],
        supports_image=True,
        supports_similarity=True,
        supports_graph=True,
    )
