from __future__ import annotations

from pydantic import BaseModel, Field


class TailPredictionRequest(BaseModel):
    head: str = Field(..., description="Head entity id, for example ent_007314")
    relation: str = Field(..., description="Relation id, for example rel_0002")
    topk: int = Field(default=5, ge=1, le=20)


class TailPredictionCandidate(BaseModel):
    entity: str
    entity_text_zh: str | None = None
    entity_text_en: str | None = None
    raw_score: float
    normalized_score: float = Field(..., ge=0.0, le=1.0)
    display_score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)


class TailPredictionResponse(BaseModel):
    task: str = "tail_prediction"
    model: str
    device: str
    inputs: TailPredictionRequest
    results: list[TailPredictionCandidate]
    latency_ms: float | None = None
