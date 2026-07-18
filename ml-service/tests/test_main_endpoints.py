"""Integration tests for FastAPI endpoints (/healthz, /embed, /ocr, /search).

Heavy ML internals are mocked; these assert wiring, auth, and response shape.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.config import Settings, get_settings

SECRET = "test-secret-key-not-real"


def _settings(**over):
    base = dict(
        ml_service_shared_secret=SECRET,
        auth_require=True,
        encoder="siglip2",
        embedding_dim_override=8,
        validate_collection_on_startup=False,
    )
    base.update(over)
    return Settings(**base)


@pytest.fixture
def client():
    main_module.app.dependency_overrides[get_settings] = lambda: _settings()
    with TestClient(main_module.app) as c:
        yield c
    main_module.app.dependency_overrides.clear()


def _png_bytes(color=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def _auth():
    return {"authorization": f"Bearer {SECRET}"}


def test_healthz_no_auth(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["encoder"] == "siglip2"


def test_search_requires_auth(client):
    resp = client.post("/search", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert resp.status_code == 401


def test_embed_requires_auth(client):
    resp = client.post("/embed", files={"files": ("x.png", _png_bytes(), "image/png")})
    assert resp.status_code == 401


def test_embed_returns_vectors(client, monkeypatch):
    class _Enc:
        dim = 8

        def embed(self, images):
            return np.tile(np.linspace(0, 1, 8, dtype=np.float32), (len(images), 1))

    monkeypatch.setattr("app.encoders.get_encoder", lambda s=None: _Enc())
    resp = client.post(
        "/embed", files={"files": ("x.png", _png_bytes(), "image/png")}, headers=_auth()
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dim"] == 8
    assert body["count"] == 1
    assert len(body["vectors"][0]) == 8


def test_search_pipeline(client, monkeypatch):
    class _Enc:
        dim = 8

        def embed(self, images):
            return np.ones((len(images), 8), dtype=np.float32)

    monkeypatch.setattr("app.encoders.get_encoder", lambda s=None: _Enc())
    monkeypatch.setattr(
        "app.search.qdrant_client.query",
        lambda vector, top_k=200, settings=None: [
            {"id": "v1", "score": 0.9, "payload": {"product_id": "p1"}},
            {"id": "v2", "score": 0.85, "payload": {"product_id": "p1"}},
            {"id": "v3", "score": 0.2, "payload": {"product_id": "p2"}},
        ],
    )
    resp = client.post(
        "/search", files={"file": ("x.png", _png_bytes(), "image/png")}, headers=_auth()
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["products"][0]["product_id"] == "p1"
    assert body["products"][0]["score"] == pytest.approx(0.9)
    assert body["weak_visual_match"] is False


def test_search_weak_match_flag(client, monkeypatch):
    class _Enc:
        dim = 8

        def embed(self, images):
            return np.ones((len(images), 8), dtype=np.float32)

    monkeypatch.setattr("app.encoders.get_encoder", lambda s=None: _Enc())
    monkeypatch.setattr(
        "app.search.qdrant_client.query",
        lambda vector, top_k=200, settings=None: [
            {"id": "v1", "score": 0.1, "payload": {"product_id": "p9"}},
        ],
    )
    resp = client.post(
        "/search", files={"file": ("x.png", _png_bytes(), "image/png")}, headers=_auth()
    )
    assert resp.json()["weak_visual_match"] is True


def test_ocr_pipeline(client, monkeypatch):
    class _Engine:
        def run(self, arr):
            return {"name": "Crocin", "strength": "650 mg", "candidates": ["Crocin 650 mg"], "tokens": []}

    monkeypatch.setattr("app.ocr.paddle.get_engine", lambda: _Engine())
    resp = client.post(
        "/ocr", files={"file": ("x.png", _png_bytes(), "image/png")}, headers=_auth()
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "Crocin"


def test_embed_no_images_400(client):
    resp = client.post("/embed", headers=_auth())
    assert resp.status_code == 400
