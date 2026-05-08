from __future__ import annotations

from fastapi import APIRouter, Query

from app.schemas.graph import SubgraphResponse
from app.services.graph_service import get_observed_subgraph

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/subgraph", response_model=SubgraphResponse, summary="Observed subgraph")
def api_get_subgraph(
    entity: str = Query(..., description="Center entity id, for example ent_007314"),
    hops: int = Query(default=1, ge=1, le=2),
    limit: int = Query(default=30, ge=1, le=100),
) -> SubgraphResponse:
    return get_observed_subgraph(entity_id=entity, hops=hops, limit=limit)
