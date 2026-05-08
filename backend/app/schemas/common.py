from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorEnvelope(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorEnvelope


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "mmkg-backend"
    model_loaded: bool = False
    dataset: str = "OpenBG-IMG"
    model: str = "Residual+Gate"
    run_dir: str | None = None
    warnings: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
