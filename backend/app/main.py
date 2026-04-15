from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.attributes import router as attributes_router
from app.routers.entities import router as entities_router
from app.routers.graph import router as graph_router
from app.routers.health import router as health_router
from app.routers.meta import router as meta_router
from app.routers.performance import router as performance_router

app = FastAPI(
    title="MMKG Backend",
    version="0.1.0",
    description="面向 MMKG Demo 前端的轻量后端 API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(meta_router)
app.include_router(performance_router)
app.include_router(entities_router)
app.include_router(attributes_router)
app.include_router(graph_router)
