from __future__ import annotations

from pydantic import BaseModel, Field


class ModelMetaResponse(BaseModel):
    model_name: str = Field(..., description="当前推理模型名")
    dataset_name: str = Field(..., description="当前推理数据集名")
    device: str = Field(..., description="当前运行设备，如 cpu/cuda")
    run_dir: str = Field(..., description="当前模型 run_dir")
    supports_image: bool = Field(..., description="当前模型是否使用图像模态")
    supports_similarity: bool = Field(default=True, description="是否支持 similar entities 查询")
    supports_graph: bool = Field(default=True, description="是否支持图谱子图查询")
