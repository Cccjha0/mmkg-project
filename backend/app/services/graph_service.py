from __future__ import annotations

from app.schemas.graph import GraphStats, SubgraphResponse


def get_observed_subgraph(
    entity_id: str,
    max_neighbors: int,
    include_relation_labels: bool,
) -> SubgraphResponse:
    """
    TODO:
    1. 从 OpenBG-IMG triples 构建邻接表
    2. 以 entity_id 为中心抽取 observed 邻接子图
    3. 结合 entity2text / relation2text 填充展示字段
    4. 统计节点数、边数、平均度
    """
    return SubgraphResponse(
        center_entity_id=entity_id,
        nodes=[],
        edges=[],
        stats=GraphStats(total_nodes=0, total_edges=0, average_degree=0.0),
    )
