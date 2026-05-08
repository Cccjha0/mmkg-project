from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EmbeddingSpace = Literal["text", "image", "fused", "entity_repr"]


class EmbeddingSummary(BaseModel):
    dim: int = Field(..., ge=1)
    l2_norm: float
    mean: float | None = None
    std: float | None = None


class EntityInfoResponse(BaseModel):
    entity: str
    entity_text: str | None = None
    entity_text_zh: str | None = None
    entity_text_en: str | None = None
    has_image: bool
    image_path: str | None = None
    image_status: Literal["available", "missing", "unknown"] = "unknown"
    available_spaces: list[EmbeddingSpace] = Field(default_factory=list)
    embedding_summary: dict[EmbeddingSpace, EmbeddingSummary] = Field(default_factory=dict)
    gate_mean: float | None = None
    model_name: str
    dataset_name: Literal["openbg_img"] = "openbg_img"


class SimilarEntityItem(BaseModel):
    entity: str
    entity_text_zh: str | None = None
    entity_text_en: str | None = None
    score: float = Field(..., ge=0.0, le=1.0)


class SimilarEntitiesResponse(BaseModel):
    task: Literal["similar"] = "similar"
    model: str
    device: str
    inputs: dict
    results: list[SimilarEntityItem]
    latency_ms: float | None = None
