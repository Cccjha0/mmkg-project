from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AttributeSource = Literal["existing", "predicted"]


class AttributeCandidate(BaseModel):
    entity: str
    entity_text_zh: str | None = None
    entity_text_en: str | None = None
    score: float | None = None
    raw_score: float | None = None
    normalized_score: float | None = None
    display_score: float | None = None
    rank: int | None = None


class AttributeRow(BaseModel):
    relation_id: str
    relation_name: str | None = None
    relation_name_en: str | None = None
    source: AttributeSource
    selected_option_index: int = Field(default=0, ge=0)
    selected_value: AttributeCandidate
    options: list[AttributeCandidate]
    candidate_count: int = Field(default=0, ge=0)
    warning: str | None = None


class AttributeEntityInfo(BaseModel):
    entity: str
    entity_text: str | None = None
    entity_text_zh: str | None = None
    entity_text_en: str | None = None
    has_image: bool
    image_path: str | None = None


class AttributeCompletionResponse(BaseModel):
    task: Literal["attribute_completion_page"] = "attribute_completion_page"
    model: str
    device: str
    inputs: dict
    results: dict
    latency_ms: float | None = None
