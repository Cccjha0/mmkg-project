from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EmbeddingSpace = Literal["text", "image", "fused", "entity_repr"]


class EmbeddingSummary(BaseModel):
    dim: int = Field(..., ge=1, description="向量维度")
    l2_norm: float = Field(..., description="L2 范数")
    mean: float | None = Field(default=None, description="均值")
    std: float | None = Field(default=None, description="标准差")


class EntityInfoResponse(BaseModel):
    entity_id: str = Field(..., description="实体 ID，例如 ent_007314")
    entity_text: str | None = Field(default=None, description="兼容字段，当前默认中文")
    entity_text_zh: str | None = Field(default=None, description="中文实体文本")
    entity_text_en: str | None = Field(default=None, description="英文实体文本")
    has_image: bool | None = Field(default=None, description="是否存在图像")
    image_path: str | None = Field(default=None, description="图像路径；无图时为 null")
    image_status: Literal["available", "missing", "unknown"] = Field(default="unknown", description="图像状态")
    available_spaces: list[EmbeddingSpace] = Field(default_factory=list, description="可用向量空间")
    embedding_summary: dict[EmbeddingSpace, EmbeddingSummary] = Field(default_factory=dict, description="各空间向量摘要")
    gate_mean: float | None = Field(default=None, description="门控平均值；模型不支持时为 null")
    model_name: str = Field(..., description="当前使用的推理模型")
    dataset_name: Literal["openbg_img"] = Field(default="openbg_img", description="当前页面固定使用 OpenBG-IMG")
