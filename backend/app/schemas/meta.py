from __future__ import annotations

from pydantic import BaseModel, Field


class RuntimeResponse(BaseModel):
    model_name: str
    model_code: str
    dataset: str
    run_dir: str
    config_path: str
    checkpoint_path: str
    device: str
    attribute_relations: list[str] = Field(default_factory=list)
    model_ready: bool = False
    warnings: list[str] = Field(default_factory=list)


class ModelMetaResponse(RuntimeResponse):
    supports_image: bool = True
    supports_similarity: bool = True
    supports_graph: bool = True
