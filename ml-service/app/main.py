"""FastAPI application for the image-search ML service.

Endpoints:
  - ``GET  /healthz`` -> liveness probe (200), reports encoder + collection.
  - ``POST /search``  -> image bytes -> embed -> ANN -> group -> threshold ->
    ``{products:[{product_id, score}], weak_visual_match}`` (Task 6).
  - ``POST /ocr``     -> image bytes -> OCR candidates + raw tokens (Task 5).
  - ``POST /embed``   -> image bytes/URL(s) -> normalized vectors + dim +
    encoder (diagnostics/backfill only, Task 4).

All non-health routes require Worker->service auth (``require_auth``). Heavy ML
imports (encoder/OCR) are done lazily inside handlers so the module imports and
``/healthz`` works without GPU/model deps.
"""
from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from .auth import require_auth
from .config import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Block startup on an encoder/collection mismatch (plan Task 2).

    Best-effort on connection errors (Qdrant may not be up yet in some deploy
    orderings), but a genuine dim/encoder mismatch raises and blocks startup.
    """
    settings = get_settings()
    if settings.validate_collection_on_startup:
        from .search import qdrant_client

        try:
            qdrant_client.validate_active_collection(settings)
        except qdrant_client.StartupValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - connection/other; don't hard-block
            logger.warning("Skipping startup collection validation: %s", exc)
    yield


app = FastAPI(
    title="Image Search ML Service",
    version="0.1.0",
    description="Self-hosted GPU embedding + OCR + vector search service.",
    lifespan=lifespan,
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


# --------------------------------------------------------------------------
# Image loading helpers
# --------------------------------------------------------------------------
def _load_image_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image bytes: {exc}")


async def _fetch_url(url: str, max_bytes: int) -> bytes:
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content = resp.content
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail="Image exceeds max size.")
        return content


def _check_size(data: bytes, settings: Settings) -> None:
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds max size.")


# --------------------------------------------------------------------------
# /embed — diagnostics / backfill only
# --------------------------------------------------------------------------
@app.post("/embed", tags=["diagnostics"], dependencies=[Depends(require_auth)])
async def embed(
    files: Optional[list[UploadFile]] = File(default=None),
    urls: Optional[str] = Form(default=None),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    """Image bytes and/or comma-separated URLs -> normalized vectors + dim + encoder.

    Diagnostics/backfill only — the production query path uses ``/search``.
    """
    from .encoders import get_encoder

    images: list[Image.Image] = []
    if files:
        for f in files:
            data = await f.read()
            _check_size(data, settings)
            images.append(_load_image_bytes(data))
    if urls:
        for url in [u.strip() for u in urls.split(",") if u.strip()]:
            data = await _fetch_url(url, settings.max_upload_bytes)
            images.append(_load_image_bytes(data))

    if not images:
        raise HTTPException(status_code=400, detail="No images provided (files or urls).")

    encoder = get_encoder(settings)
    vectors = encoder.embed(images)
    return JSONResponse(
        content={
            "encoder": settings.encoder,
            "dim": encoder.dim,
            "count": int(vectors.shape[0]),
            "vectors": vectors.tolist(),
        }
    )


# --------------------------------------------------------------------------
# /ocr — candidate query strings + raw tokens for the existing text search
# --------------------------------------------------------------------------
@app.post("/ocr", tags=["ocr"], dependencies=[Depends(require_auth)])
async def ocr(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    """Image bytes -> OCR candidate query string(s) + raw tokens."""
    import numpy as np

    from .ocr.paddle import get_engine

    data = await file.read()
    _check_size(data, settings)
    image = _load_image_bytes(data)
    # PaddleOCR expects an ndarray (or path); pass an RGB ndarray.
    arr = np.asarray(image.convert("RGB"))
    result = get_engine().run(arr)
    return JSONResponse(content=result)


# --------------------------------------------------------------------------
# /search — image -> product IDs (embeds internally)
# --------------------------------------------------------------------------
@app.post("/search", tags=["search"], dependencies=[Depends(require_auth)])
async def search(
    file: UploadFile = File(...),
    top_k: int = Form(default=200),
    pool: str = Form(default="max"),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    """Image bytes -> embed -> ANN -> group -> threshold.

    Returns ``{products:[{product_id, score}], weak_visual_match, encoder}``.
    """
    from .encoders import get_encoder
    from .search import qdrant_client
    from .search.grouping import group_by_product
    from .search.threshold import apply_threshold

    data = await file.read()
    _check_size(data, settings)
    image = _load_image_bytes(data)

    encoder = get_encoder(settings)
    vectors = encoder.embed([image])
    hits = qdrant_client.query(vectors[0], top_k=top_k, settings=settings)
    products = group_by_product(hits, pool=pool)
    weak = apply_threshold(products, encoder=settings.encoder)

    return JSONResponse(
        content={
            "encoder": settings.encoder,
            "products": [
                {"product_id": p["product_id"], "score": p["score"]} for p in products
            ],
            "weak_visual_match": weak,
        }
    )
