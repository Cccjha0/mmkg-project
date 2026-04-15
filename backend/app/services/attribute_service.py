from __future__ import annotations

from app.deps import get_runtime_config
from app.schemas.attribute_completion import AttributeCompletionResponse


def get_attribute_completion(entity_id: str, topk: int) -> AttributeCompletionResponse:
    """
    TODO:
    1. 加载一组 attribute_relations
    2. 先查数据集中该实体在这些关系上已有的 tail，标记为 existing
    3. 对缺失关系调用 predictor.complete_attributes / predictor.predict_tail
    4. predicted 行返回 top-k candidates
    5. selected_value 默认取 candidates[0]
    """
    cfg = get_runtime_config()
    return AttributeCompletionResponse(
        entity_id=entity_id,
        model_name=cfg["model_name"],
        dataset_name="openbg_img",
        rows=[],
    )
