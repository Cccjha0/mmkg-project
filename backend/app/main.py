from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers.attributes import router as attributes_router
from app.routers.entities import router as entities_router
from app.routers.graph import router as graph_router
from app.routers.health import router as health_router
from app.routers.meta import router as meta_router
from app.routers.performance import router as performance_router
from app.routers.predict import router as predict_router
from app.deps import repo_root

app = FastAPI(
    title="MMKG Backend",
    version="0.1.0",
    description="Backend API for the MMKG demo.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static/openbg_img",
    StaticFiles(directory=repo_root() / "data" / "datasets" / "openbg_img" / "raw" / "OpenBG-IMG_images"),
    name="openbg-img-static",
)

app.include_router(health_router)
app.include_router(meta_router)
app.include_router(performance_router)
app.include_router(entities_router)
app.include_router(attributes_router)
app.include_router(graph_router)
app.include_router(predict_router)
