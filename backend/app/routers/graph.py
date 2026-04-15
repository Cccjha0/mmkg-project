from __future__ import annotations

from fastapi import APIRouter, Query

from app.schemas.graph import SubgraphResponse
from app.services.graph_service import get_observed_subgraph

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/subgraph", response_model=SubgraphResponse, summary="获取 observed 子图")
def api_get_subgraph(
    entity_id: str = Query(..., description="中心实体 ID，例如 ent_007314"),
    max_neighbors: int = Query(default=20, ge=1, le=100, description="最多返回的邻居数"),
    include_relation_labels: bool = Query(default=True, description="是否返回关系文本"),
) -> SubgraphResponse:
    return get_observed_subgraph(
        entity_id=entity_id,
        max_neighbors=max_neighbors,
        include_relation_labels=include_relation_labels,
    )
