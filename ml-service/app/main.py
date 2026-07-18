"""FastAPI application for the image-search ML service.

This module wires the app together and exposes:
  - ``GET  /healthz``  -> liveness probe (200), fully implemented here.
  - ``POST /search``   -> image -> product IDs (STUB, owned by Task 6).
  - ``POST /ocr``      -> OCR candidates (STUB, owned by Task 5).
  - ``POST /embed``    -> diagnostics/backfill embeddings (STUB, owned by Task 4).

The stub routes are guarded by the real auth dependency (``require_auth``) and
return HTTP 501 with a documented payload until the owning agents fill in the
business logic. The app imports cleanly and boots without GPU/model deps.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from .auth import require_auth
from .config import Settings, get_settings

app = FastAPI(
    title="Image Search ML Service",
    version="0.1.0",
    description="Self-hosted GPU embedding + OCR + vector search service.",
)


@app.get("/healthz", tags=["ops"])
async def healthz(settings: Settings = Depends(get_settings)) -> dict:
    """Liveness/readiness probe. Reports active encoder + collection."""
    return {
        "status": "ok",
        "encoder": settings.encoder,
        "embedding_dim": settings.embedding_dim,
        "active_collection": settings.active_collection,
    }


def _not_implemented(endpoint: str, owner: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "status": "not_implemented",
            "endpoint": endpoint,
            "detail": f"{endpoint} is a stub; implemented by {owner}.",
        },
    )


@app.post("/search", tags=["search"], dependencies=[Depends(require_auth)])
async def search() -> JSONResponse:
    """STUB (Task 6): image bytes -> {products:[{product_id, score}], weak_visual_match}."""
    return _not_implemented("/search", "Task 6 (query/search service)")


@app.post("/ocr", tags=["ocr"], dependencies=[Depends(require_auth)])
async def ocr() -> JSONResponse:
    """STUB (Task 5): image bytes -> {candidates:[...], tokens:[...]}."""
    return _not_implemented("/ocr", "Task 5 (OCR pipeline)")


@app.post("/embed", tags=["diagnostics"], dependencies=[Depends(require_auth)])
async def embed() -> JSONResponse:
    """STUB (Task 4): image bytes/URL -> {vectors:[...], dim, encoder}. Diagnostics/backfill only."""
    return _not_implemented("/embed", "Task 4 (embedding service)")
