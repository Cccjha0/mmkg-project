from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AttributeSource = Literal["existing", "predicted"]


class AttributeCandidate(BaseModel):
    entity_id: str = Field(..., description="候选值实体 ID")
    entity_text: str | None = Field(default=None, description="兼容字段，当前默认中文")
    entity_text_zh: str | None = Field(default=None, description="中文文本")
    entity_text_en: str | None = Field(default=None, description="英文文本")
    score: float = Field(..., description="模型得分")


class AttributeRow(BaseModel):
    relation_id: str = Field(..., description="关系 ID，例如 rel_0001")
    relation_text: str | None = Field(default=None, description="兼容字段，当前默认中文")
    relation_text_zh: str | None = Field(default=None, description="中文关系文本")
    relation_text_en: str | None = Field(default=None, description="英文关系文本")
    source: AttributeSource = Field(..., description="属性来源：existing 或 predicted")
    selected_value: AttributeCandidate | None = Field(default=None, description="当前展示值")
    candidates: list[AttributeCandidate] = Field(default_factory=list, description="候选列表；existing 时通常只有 1 项，predicted 时建议返回 top-k")
    selected_index: int = Field(default=0, ge=0, description="前端默认选中的候选下标")


class AttributeCompletionQuery(BaseModel):
    topk: int = Field(default=5, ge=1, le=20, description="predicted 属性返回的候选数量")


class AttributeCompletionResponse(BaseModel):
    entity_id: str = Field(..., description="当前商品实体 ID")
    model_name: str = Field(..., description="当前固定使用的推理模型名")
    dataset_name: Literal["openbg_img"] = Field(default="openbg_img", description="当前页面固定数据集")
    rows: list[AttributeRow] = Field(default_factory=list, description="属性表行数据")
