from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    label: str
    label_zh: str | None = None
    label_en: str | None = None
    type: str | None = None
    has_image: bool | None = None


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    relation_text_zh: str | None = None
    relation_text_en: str | None = None
    kind: Literal["observed", "predicted"] = "observed"
    score: float | None = None


class GraphCenter(BaseModel):
    id: str
    label_zh: str | None = None
    label_en: str | None = None
    has_image: bool


class GraphStats(BaseModel):
    num_nodes: int = Field(..., ge=0)
    num_edges: int = Field(..., ge=0)


class SubgraphResponse(BaseModel):
    center: GraphCenter
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    stats: GraphStats
