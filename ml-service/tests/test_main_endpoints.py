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
        "/search",
        content=_png_bytes(),
        headers={**_auth(), "content-type": "application/octet-stream"},
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
        "/search",
        content=_png_bytes(),
        headers={**_auth(), "content-type": "application/octet-stream"},
    )
    assert resp.json()["weak_visual_match"] is True


def test_ocr_pipeline(client, monkeypatch):
    class _Engine:
        def run(self, arr):
            return {"name": "Crocin", "strength": "650 mg", "candidates": ["Crocin 650 mg"], "tokens": []}

    monkeypatch.setattr("app.ocr.paddle.get_engine", lambda: _Engine())
    resp = client.post(
        "/ocr",
        content=_png_bytes(),
        headers={**_auth(), "content-type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "Crocin"


def test_embed_no_images_400(client):
    resp = client.post("/embed", headers=_auth())
    assert resp.status_code == 400


# --------------------------------------------------------------------------
# Cross-service contract: send EXACTLY what worker/src/ml_client.ts sends —
# raw octet-stream body + HMAC (X-Timestamp / X-Signature) headers. No multipart.
# --------------------------------------------------------------------------
def _hmac_headers(method: str, path: str):
    import time

    from app.auth import compute_signature

    ts = str(int(time.time()))
    sig = compute_signature(SECRET, ts, method, path)
    return {
        "x-timestamp": ts,
        "x-signature": sig,
        "content-type": "application/octet-stream",
    }


def test_search_contract_raw_bytes_hmac(client, monkeypatch):
    class _Enc:
        dim = 8

        def embed(self, images):
            return np.ones((len(images), 8), dtype=np.float32)

    monkeypatch.setattr("app.encoders.get_encoder", lambda s=None: _Enc())
    monkeypatch.setattr(
        "app.search.qdrant_client.query",
        lambda vector, top_k=200, settings=None: [
            {"id": "v1", "score": 0.9, "payload": {"product_id": "p1"}},
        ],
    )
    # Exactly the Worker's request: raw bytes body + HMAC headers, no multipart.
    resp = client.post(
        "/search", content=_png_bytes(), headers=_hmac_headers("POST", "/search")
    )
    assert resp.status_code == 200
    assert resp.json()["products"][0]["product_id"] == "p1"


def test_ocr_contract_raw_bytes_hmac(client, monkeypatch):
    class _Engine:
        def run(self, arr):
            return {"name": "Crocin", "strength": "650 mg", "candidates": [], "tokens": []}

    monkeypatch.setattr("app.ocr.paddle.get_engine", lambda: _Engine())
    resp = client.post(
        "/ocr", content=_png_bytes(), headers=_hmac_headers("POST", "/ocr")
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "Crocin"


def test_search_empty_body_400(client):
    resp = client.post(
        "/search", content=b"", headers={**_auth(), "content-type": "application/octet-stream"}
    )
    assert resp.status_code == 400


def test_ocr_empty_body_400(client):
    resp = client.post(
        "/ocr", content=b"", headers={**_auth(), "content-type": "application/octet-stream"}
    )
    assert resp.status_code == 400


# --------------------------------------------------------------------------
# Calibrated threshold file: present -> calibrated value used; absent -> default.
# --------------------------------------------------------------------------
def _search_with_score(client, monkeypatch, score):
    class _Enc:
        dim = 8

        def embed(self, images):
            return np.ones((len(images), 8), dtype=np.float32)

    monkeypatch.setattr("app.encoders.get_encoder", lambda s=None: _Enc())
    monkeypatch.setattr(
        "app.search.qdrant_client.query",
        lambda vector, top_k=200, settings=None: [
            {"id": "v1", "score": score, "payload": {"product_id": "p1"}},
        ],
    )
    resp = client.post(
        "/search", content=_png_bytes(), headers={**_auth(), "content-type": "application/octet-stream"}
    )
    return resp.json()["weak_visual_match"]


def test_calibrated_threshold_file_used(tmp_path, monkeypatch):
    import json

    # Calibrated threshold 0.8 for siglip2 (much higher than provisional 0.35).
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"encoders": {"siglip2": {"weak_visual_match_threshold": 0.8}}}))

    tuned = _settings(weak_match_threshold_file=str(path))
    # lifespan loads the file via get_settings() directly, so patch both the
    # startup loader's settings and the request-time dependency.
    monkeypatch.setattr(main_module, "get_settings", lambda: tuned)
    main_module.app.dependency_overrides[get_settings] = lambda: tuned
    with TestClient(main_module.app) as c:
        # Score 0.5: below calibrated 0.8 -> weak; would be strong under 0.35.
        assert _search_with_score(c, monkeypatch, 0.5) is True
        # Score 0.9: above calibrated 0.8 -> strong.
        assert _search_with_score(c, monkeypatch, 0.9) is False
    main_module.app.dependency_overrides.clear()


def test_provisional_default_when_no_file(monkeypatch):
    tuned = _settings(weak_match_threshold_file=None)
    monkeypatch.setattr(main_module, "get_settings", lambda: tuned)
    main_module.app.dependency_overrides[get_settings] = lambda: tuned
    with TestClient(main_module.app) as c:
        # Provisional default 0.35: 0.5 is strong, 0.1 is weak.
        assert _search_with_score(c, monkeypatch, 0.5) is False
        assert _search_with_score(c, monkeypatch, 0.1) is True
    main_module.app.dependency_overrides.clear()
