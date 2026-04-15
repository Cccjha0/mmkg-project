from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


NodeType = Literal["product", "attribute", "brand", "category", "material", "color", "unknown"]
EdgeKind = Literal["observed"]


class GraphNode(BaseModel):
    id: str = Field(..., description="节点实体 ID")
    label: str = Field(..., description="节点展示名")
    label_zh: str | None = Field(default=None, description="中文文本")
    label_en: str | None = Field(default=None, description="英文文本")
    type: NodeType = Field(default="unknown", description="节点类型")
    has_image: bool | None = Field(default=None, description="是否有图像")
    is_center: bool = Field(default=False, description="是否为中心节点")


class GraphEdge(BaseModel):
    id: str = Field(..., description="边唯一 ID")
    source: str = Field(..., description="源节点 ID")
    target: str = Field(..., description="目标节点 ID")
    relation_id: str = Field(..., description="关系 ID")
    relation_text: str | None = Field(default=None, description="兼容字段，当前默认中文")
    relation_text_zh: str | None = Field(default=None, description="中文关系文本")
    relation_text_en: str | None = Field(default=None, description="英文关系文本")
    kind: EdgeKind = Field(default="observed", description="当前第一版仅支持 observed")


class GraphStats(BaseModel):
    total_nodes: int = Field(..., ge=0, description="节点总数")
    total_edges: int = Field(..., ge=0, description="边总数")
    average_degree: float = Field(..., ge=0.0, description="平均度数")


class SubgraphQuery(BaseModel):
    entity_id: str = Field(..., description="中心实体 ID")
    max_neighbors: int = Field(default=20, ge=1, le=100, description="最多返回的邻居数")
    include_relation_labels: bool = Field(default=True, description="是否返回关系文本")


class SubgraphResponse(BaseModel):
    dataset_name: Literal["openbg_img"] = Field(default="openbg_img", description="当前图谱页面固定数据集")
    center_entity_id: str = Field(..., description="中心实体 ID")
    nodes: list[GraphNode] = Field(default_factory=list, description="子图节点列表")
    edges: list[GraphEdge] = Field(default_factory=list, description="子图边列表")
    stats: GraphStats = Field(..., description="子图统计信息")
