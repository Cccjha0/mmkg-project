from __future__ import annotations

from app.deps import get_runtime_config
from app.schemas.entity import EntityInfoResponse


def get_entity_info(entity_id: str) -> EntityInfoResponse:
    """
    TODO:
    1. 初始化/缓存 predictor
    2. 调用 predictor.get_entity_multimodal(entity_id)
    3. 将原始输出映射到 EntityInfoResponse
    """
    cfg = get_runtime_config()
    return EntityInfoResponse(
        entity_id=entity_id,
        entity_text=None,
        entity_text_zh=None,
        entity_text_en=None,
        has_image=None,
        image_path=None,
        image_status="unknown",
        available_spaces=[],
        embedding_summary={},
        gate_mean=None,
        model_name=cfg["model_name"],
        dataset_name="openbg_img",
    )
