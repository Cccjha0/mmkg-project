from __future__ import annotations

from app.schemas.graph import GraphCenter, GraphEdge, GraphNode, GraphStats, SubgraphResponse
from app.services.entity_service import _ensure_entity
from app.services.openbg_img_data import (
    entity_text,
    entity_text_en,
    image_path_for_entity,
    relation_text,
    relation_text_en,
    triples_by_entity,
)


def _node(entity: str, node_type: str, center: bool = False) -> GraphNode:
    text = entity_text(entity)
    text_en = entity_text_en(entity)
    return GraphNode(
        id=entity,
        label=text or entity,
        label_zh=text,
        label_en=text_en,
        type="demo_product" if center else node_type,
        has_image=image_path_for_entity(entity) is not None,
    )


def get_observed_subgraph(entity_id: str, hops: int, limit: int) -> SubgraphResponse:
    _ensure_entity(entity_id)
    center_text = entity_text(entity_id)
    center_text_en = entity_text_en(entity_id)
    center = GraphCenter(
        id=entity_id,
        label_zh=center_text,
        label_en=center_text_en,
        has_image=image_path_for_entity(entity_id) is not None,
    )
    nodes: dict[str, GraphNode] = {entity_id: _node(entity_id, "entity", center=True)}
    edges: list[GraphEdge] = []
    frontier = {entity_id}
    visited = {entity_id}

    for _ in range(hops):
        next_frontier: set[str] = set()
        for current in sorted(frontier):
            for head, relation, tail in triples_by_entity().get(current, []):
                if len(edges) >= limit:
                    break
                source = head
                target = tail
                for entity in (source, target):
                    if entity not in nodes and len(nodes) < limit + 1:
                        node_type = "attribute_value" if source == entity_id and entity == target else "entity"
                        nodes[entity] = _node(entity, node_type)
                    if entity not in visited:
                        next_frontier.add(entity)
                        visited.add(entity)
                rel_text = relation_text(relation)
                rel_text_en = relation_text_en(relation)
                edges.append(
                    GraphEdge(
                        id=f"e_{len(edges) + 1}",
                        source=source,
                        target=target,
                        relation=relation,
                        relation_text_zh=rel_text,
                        relation_text_en=rel_text_en,
                        kind="observed",
                        score=None,
                    )
                )
            if len(edges) >= limit:
                break
        frontier = next_frontier
        if not frontier or len(edges) >= limit:
            break

    return SubgraphResponse(
        center=center,
        nodes=list(nodes.values()),
        edges=edges,
        stats=GraphStats(num_nodes=len(nodes), num_edges=len(edges)),
    )
