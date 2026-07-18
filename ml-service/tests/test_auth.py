"""Unit tests for Worker->service authentication (app/auth.py)."""
from __future__ import annotations

import time

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.auth import compute_signature, require_auth, verify_request
from app.config import Settings

SECRET = "test-secret-key-not-real"


def make_settings(**overrides) -> Settings:
    base = dict(
        ml_service_shared_secret=SECRET,
        auth_require=True,
        auth_replay_window_seconds=300,
    )
    base.update(overrides)
    return Settings(**base)


def make_request(method: str, path: str, headers: dict[str, str]) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "headers": raw_headers,
        "query_string": b"",
    }
    return Request(scope)


# ---- verify_request unit tests (no server) -----------------------------

def test_valid_bearer_passes():
    s = make_settings()
    req = make_request("POST", "/search", {"authorization": f"Bearer {SECRET}"})
    assert verify_request(req, s) is True


def test_wrong_bearer_fails():
    s = make_settings()
    req = make_request("POST", "/search", {"authorization": "Bearer wrong"})
    assert verify_request(req, s) is False


def test_missing_auth_fails():
    s = make_settings()
    req = make_request("POST", "/search", {})
    assert verify_request(req, s) is False


def test_valid_hmac_passes():
    s = make_settings()
    ts = str(int(time.time()))
    sig = compute_signature(SECRET, ts, "POST", "/search")
    req = make_request("POST", "/search", {"x-signature": sig, "x-timestamp": ts})
    assert verify_request(req, s) is True


def test_hmac_wrong_signature_fails():
    s = make_settings()
    ts = str(int(time.time()))
    req = make_request("POST", "/search", {"x-signature": "deadbeef", "x-timestamp": ts})
    assert verify_request(req, s) is False


def test_hmac_expired_timestamp_fails():
    s = make_settings(auth_replay_window_seconds=60)
    ts = str(int(time.time()) - 120)  # outside window
    sig = compute_signature(SECRET, ts, "POST", "/search")
    req = make_request("POST", "/search", {"x-signature": sig, "x-timestamp": ts})
    assert verify_request(req, s) is False


def test_hmac_signature_bound_to_path():
    s = make_settings()
    ts = str(int(time.time()))
    sig = compute_signature(SECRET, ts, "POST", "/search")
    # Same signature replayed against a different path must fail.
    req = make_request("POST", "/ocr", {"x-signature": sig, "x-timestamp": ts})
    assert verify_request(req, s) is False


def test_auth_disabled_bypasses():
    s = make_settings(auth_require=False)
    req = make_request("POST", "/search", {})
    assert verify_request(req, s) is True


def test_no_key_configured_fails_closed():
    s = make_settings(ml_service_shared_secret=None)
    req = make_request("POST", "/search", {"authorization": "Bearer anything"})
    assert verify_request(req, s) is False


# ---- FastAPI dependency integration ------------------------------------

def _app_with_settings(settings: Settings) -> TestClient:
    app = FastAPI()

    from app.config import get_settings

    app.dependency_overrides[get_settings] = lambda: settings

    @app.post("/protected", dependencies=[Depends(require_auth)])
    async def protected():
        return {"ok": True}

    return TestClient(app)


def test_dependency_401_without_auth():
    client = _app_with_settings(make_settings())
    resp = client.post("/protected")
    assert resp.status_code == 401


def test_dependency_200_with_bearer():
    client = _app_with_settings(make_settings())
    resp = client.post("/protected", headers={"authorization": f"Bearer {SECRET}"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_dependency_200_with_hmac():
    client = _app_with_settings(make_settings())
    ts = str(int(time.time()))
    sig = compute_signature(SECRET, ts, "POST", "/protected")
    resp = client.post("/protected", headers={"x-signature": sig, "x-timestamp": ts})
    assert resp.status_code == 200
