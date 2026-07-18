"""Tests for app/d1_client.py — the D1 REST client.

A fake httpx-style session captures requests and returns canned Cloudflare D1
envelopes, so parameterized query/execute/batch/paginate logic is exercised
without any network. Credential validation is also covered.
"""
from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.d1_client import D1Client, D1ConfigError, D1RequestError


class _FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text or json.dumps(self._body)

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def post(self, url, content=None):
        self.requests.append({"url": url, "body": json.loads(content)})
        return self._responses.pop(0)


def _settings():
    return Settings(d1_account_id="acct", d1_database_id="db", d1_api_token="tok")


def _ok(results=None, meta=None):
    return _FakeResponse(body={
        "success": True,
        "result": [{"results": results or [], "meta": meta or {}, "success": True}],
    })


def test_query_returns_rows():
    session = _FakeSession([_ok(results=[{"product_id": "p1"}, {"product_id": "p2"}])])
    db = D1Client(_settings(), session=session)
    rows = db.query("SELECT product_id FROM products WHERE active = ?", [1])
    assert rows == [{"product_id": "p1"}, {"product_id": "p2"}]
    # Parameterized: sql + params sent verbatim.
    assert session.requests[0]["body"]["sql"].startswith("SELECT product_id")
    assert session.requests[0]["body"]["params"] == [1]


def test_execute_returns_meta():
    session = _FakeSession([_ok(meta={"changes": 1, "last_row_id": 5})])
    db = D1Client(_settings(), session=session)
    meta = db.execute("INSERT INTO products (product_id, name) VALUES (?, ?)", ["p1", "n"])
    assert meta["changes"] == 1


def test_batch_runs_all_statements():
    session = _FakeSession([_ok(meta={"changes": 1}), _ok(meta={"changes": 1})])
    db = D1Client(_settings(), session=session)
    metas = db.batch([
        ("UPDATE products SET active = 0 WHERE product_id = ?", ["p1"]),
        ("UPDATE products SET active = 0 WHERE product_id = ?", ["p2"]),
    ])
    assert len(metas) == 2
    assert len(session.requests) == 2


def test_paginate_appends_limit_offset_and_stops():
    # First page full (2 rows), second page short (1 row) -> stop.
    session = _FakeSession([
        _ok(results=[{"image_id": "i1"}, {"image_id": "i2"}]),
        _ok(results=[{"image_id": "i3"}]),
    ])
    db = D1Client(_settings(), session=session)
    rows = list(db.paginate("SELECT image_id FROM product_images ORDER BY image_id", page_size=2))
    assert [r["image_id"] for r in rows] == ["i1", "i2", "i3"]
    # LIMIT ? OFFSET ? appended with the page params.
    assert session.requests[0]["body"]["sql"].endswith("LIMIT ? OFFSET ?")
    assert session.requests[0]["body"]["params"] == [2, 0]
    assert session.requests[1]["body"]["params"] == [2, 2]


def test_missing_creds_raises():
    db = D1Client(Settings(d1_account_id=None, d1_database_id=None, d1_api_token=None),
                  session=_FakeSession([]))
    with pytest.raises(D1ConfigError):
        db.query("SELECT 1")


def test_http_error_raises():
    session = _FakeSession([_FakeResponse(status_code=500, text="boom")])
    db = D1Client(_settings(), session=session)
    with pytest.raises(D1RequestError):
        db.query("SELECT 1")


def test_cf_envelope_failure_raises():
    session = _FakeSession([_FakeResponse(body={"success": False, "errors": [{"message": "bad"}]})])
    db = D1Client(_settings(), session=session)
    with pytest.raises(D1RequestError):
        db.execute("INSERT INTO products (product_id, name) VALUES (?, ?)", ["p", "n"])
