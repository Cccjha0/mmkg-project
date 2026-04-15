from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    code: str = Field(..., description="错误码")
    message: str = Field(..., description="错误信息")
    detail: str | None = Field(default=None, description="附加错误详情")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "mmkg-backend"
    version: str = "0.1.0"
    model_loaded: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
