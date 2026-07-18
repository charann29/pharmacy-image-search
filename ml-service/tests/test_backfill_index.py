"""Tests for scripts/backfill_index.py.

Strategy (no GPU / no model / no network):
  * D1 -> SQLite-backed fake with 0001_init.sql applied.
  * Qdrant -> in-memory ``QdrantClient(":memory:")`` (real client logic).
  * Encoder -> deterministic fake: each distinct image maps to a stable, unit
    vector derived from a hash of its pixels, so a known image's nearest
    neighbour is itself.
  * ImageKit fetch -> mock returning a PNG whose pixels encode the image_id.

Covers: dry-run indexes all (100-image fixture), second run indexes 0
(idempotent), kill mid-run + resume reaches full count with no dup ids, Qdrant
count == mapped count, and known image's NN == itself.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest

from app.config import Settings
from app.search import qdrant_client as qc

from _fake_d1 import FakeD1Client
from conftest import requires_qdrant

from scripts import backfill_index as bi

DIM = 16


# --------------------------------------------------------------------------
# Deterministic fake encoder + fixtures
# --------------------------------------------------------------------------
class FakeEncoder:
    """Maps each distinct image to a stable unit vector from its pixel hash."""

    def __init__(self, dim=DIM):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def embed(self, images):
        out = np.zeros((len(images), self._dim), dtype=np.float32)
        for i, img in enumerate(images):
            h = hashlib.sha256(img.tobytes()).digest()
            seed = int.from_bytes(h[:8], "big")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._dim).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-12)
        return out


def _png_for(image_id: str) -> bytes:
    """Build a small PNG whose pixel content is unique per image_id."""
    from PIL import Image

    h = hashlib.sha256(image_id.encode()).digest()
    color = (h[0], h[1], h[2])
    img = Image.new("RGB", (8, 8), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockFetcher:
    def __init__(self):
        self.calls = 0

    def __call__(self, url: str) -> bytes:
        self.calls += 1
        # url ends with the transform param; recover the image_id from our map.
        # We stash image_id in the url query below.
        image_id = url.split("imgid=")[1].split("&")[0]
        return _png_for(image_id)


def _seed_images(db: FakeD1Client, n: int):
    # product per 4 images.
    stmts = []
    for k in range((n + 3) // 4):
        pid = f"p{k}"
        stmts.append((
            "INSERT INTO products (product_id, name, active) VALUES (?, ?, 1)",
            [pid, f"prod {k}"],
        ))
    for i in range(n):
        iid = f"img{i:04d}"
        pid = f"p{i // 4}"
        # encode image_id into the url so the mock fetcher can recover it.
        url = f"https://ik/base?imgid={iid}"
        stmts.append((
            "INSERT INTO product_images (image_id, product_id, imagekit_file_id, imagekit_url, "
            "is_reference, source_updated_at, deleted_at, created_at) "
            "VALUES (?, ?, ?, ?, 0, '2026-01-01T00:00:00Z', NULL, '2026-01-01T00:00:00Z')",
            [iid, pid, f"f{i}", url],
        ))
    db.batch(stmts)


def _settings():
    return Settings(encoder="siglip2", embedding_dim_override=DIM,
                    validate_collection_on_startup=False)


def _new_qdrant():
    from qdrant_client import QdrantClient

    return QdrantClient(":memory:")


# --------------------------------------------------------------------------
# transform url
# --------------------------------------------------------------------------
def test_build_transform_url():
    assert bi.build_transform_url("https://ik/x", 512) == "https://ik/x?tr=w-512"
    assert bi.build_transform_url("https://ik/x?a=1", 512) == "https://ik/x?a=1&tr=w-512"
    assert bi.build_transform_url("", 512) == ""


def test_retry_with_backoff_gives_up():
    calls = {"n": 0}

    def failing():
        calls["n"] += 1
        raise bi.TransientFetchError("boom")

    with pytest.raises(bi.TransientFetchError):
        bi.retry_with_backoff(failing, max_retries=3, sleep=lambda s: None)
    assert calls["n"] == 4  # 1 + 3 retries


def test_retry_with_backoff_succeeds_after_transient():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise bi.TransientFetchError("transient")
        return b"ok"

    assert bi.retry_with_backoff(flaky, max_retries=5, sleep=lambda s: None) == b"ok"


# --------------------------------------------------------------------------
# Backfill integration (in-memory Qdrant)
# --------------------------------------------------------------------------
@requires_qdrant
def test_backfill_indexes_all_100():
    db = FakeD1Client()
    _seed_images(db, 100)
    settings = _settings()
    qdrant = _new_qdrant()
    name = settings.active_collection

    stats = bi.run_backfill(
        db=db, encoder=FakeEncoder(), settings=settings, qdrant=qdrant,
        image_fetcher=MockFetcher(), collection_name=name, batch_size=16,
    )
    assert stats.indexed == 100
    # Qdrant count == mapped count.
    assert qdrant.count(name).count == 100
    assert db.query("SELECT COUNT(*) c FROM embedding_map WHERE encoder='siglip2'")[0]["c"] == 100
    assert stats.images_per_sec >= 0


@requires_qdrant
def test_backfill_second_run_indexes_zero():
    db = FakeD1Client()
    _seed_images(db, 20)
    settings = _settings()
    qdrant = _new_qdrant()
    name = settings.active_collection

    bi.run_backfill(db=db, encoder=FakeEncoder(), settings=settings, qdrant=qdrant,
                    image_fetcher=MockFetcher(), collection_name=name, batch_size=8)
    stats2 = bi.run_backfill(db=db, encoder=FakeEncoder(), settings=settings, qdrant=qdrant,
                             image_fetcher=MockFetcher(), collection_name=name, batch_size=8)
    assert stats2.indexed == 0
    assert stats2.skipped_existing == 20
    assert qdrant.count(name).count == 20


@requires_qdrant
def test_backfill_kill_and_resume(tmp_path):
    db = FakeD1Client()
    _seed_images(db, 50)
    settings = _settings()
    qdrant = _new_qdrant()
    name = settings.active_collection
    ckpt_path = str(tmp_path / "ckpt.json")

    # First run: simulate a kill after ~2 batches by raising inside the fetcher.
    fetcher = MockFetcher()
    killer = {"n": 0}

    def killing_fetch(url):
        killer["n"] += 1
        if killer["n"] > 20:
            raise KeyboardInterrupt("simulated kill")
        return fetcher(url)

    ckpt = bi.Checkpoint(path=ckpt_path).load()
    with pytest.raises(KeyboardInterrupt):
        bi.run_backfill(db=db, encoder=FakeEncoder(), settings=settings, qdrant=qdrant,
                        image_fetcher=killing_fetch, collection_name=name,
                        batch_size=10, checkpoint=ckpt)
    partial = qdrant.count(name).count
    assert 0 < partial < 50

    # Resume with a fresh checkpoint load + working fetcher.
    ckpt2 = bi.Checkpoint(path=ckpt_path).load()
    assert ckpt2.last_image_id is not None
    stats = bi.run_backfill(db=db, encoder=FakeEncoder(), settings=settings, qdrant=qdrant,
                            image_fetcher=MockFetcher(), collection_name=name,
                            batch_size=10, checkpoint=ckpt2)
    # Full count reached, no duplicate ids (count == 50 exactly).
    assert qdrant.count(name).count == 50
    assert db.query("SELECT COUNT(*) c FROM embedding_map")[0]["c"] == 50
    assert db.query("SELECT COUNT(DISTINCT vector_id) c FROM embedding_map")[0]["c"] == 50


@requires_qdrant
def test_known_image_nearest_neighbour_is_itself():
    db = FakeD1Client()
    _seed_images(db, 30)
    settings = _settings()
    qdrant = _new_qdrant()
    name = settings.active_collection
    enc = FakeEncoder()

    bi.run_backfill(db=db, encoder=enc, settings=settings, qdrant=qdrant,
                    image_fetcher=MockFetcher(), collection_name=name, batch_size=8)

    # Re-embed a known image the same way the backfill did, then query.
    from PIL import Image
    target = "img0007"
    img = Image.open(io.BytesIO(_png_for(target))).convert("RGB")
    vec = enc.embed([img])[0]
    hits = qc.query(vec, top_k=1, settings=settings, client=qdrant, collection_name=name)
    assert hits[0]["payload"]["image_id"] == target


@requires_qdrant
def test_backfill_dry_run_writes_nothing():
    db = FakeD1Client()
    _seed_images(db, 10)
    settings = _settings()
    qdrant = _new_qdrant()
    name = settings.active_collection

    stats = bi.run_backfill(db=db, encoder=FakeEncoder(), settings=settings, qdrant=qdrant,
                            image_fetcher=MockFetcher(), collection_name=name,
                            batch_size=4, dry_run=True)
    assert stats.indexed == 10  # counted as processed
    assert qdrant.count(name).count == 0
    assert db.query("SELECT COUNT(*) c FROM embedding_map")[0]["c"] == 0


@requires_qdrant
def test_backfill_skips_soft_deleted():
    db = FakeD1Client()
    _seed_images(db, 10)
    db.execute("UPDATE product_images SET deleted_at = '2026-03-01T00:00:00Z' WHERE image_id = 'img0000'")
    settings = _settings()
    qdrant = _new_qdrant()
    name = settings.active_collection
    stats = bi.run_backfill(db=db, encoder=FakeEncoder(), settings=settings, qdrant=qdrant,
                            image_fetcher=MockFetcher(), collection_name=name, batch_size=4)
    assert stats.indexed == 9
    assert qdrant.count(name).count == 9


@requires_qdrant
def test_dry_run_checkpoint_does_not_poison_real_run(tmp_path):
    """Review #2: a dry-run must not advance the persistent checkpoint, so a
    subsequent real run still indexes every row."""
    db = FakeD1Client()
    _seed_images(db, 30)
    settings = _settings()
    qdrant = _new_qdrant()
    name = settings.active_collection
    ckpt_path = str(tmp_path / "ckpt.json")

    # Dry-run with a checkpoint present.
    ckpt = bi.Checkpoint(path=ckpt_path).load()
    dry_stats = bi.run_backfill(db=db, encoder=FakeEncoder(), settings=settings, qdrant=qdrant,
                                image_fetcher=MockFetcher(), collection_name=name,
                                batch_size=8, checkpoint=ckpt, dry_run=True)
    assert dry_stats.indexed == 30
    # Checkpoint file must NOT have been written/advanced by the dry-run.
    import os
    assert not os.path.exists(ckpt_path)

    # Real run resumes from an unpolluted checkpoint and indexes all rows.
    ckpt2 = bi.Checkpoint(path=ckpt_path).load()
    assert ckpt2.last_image_id is None
    real_stats = bi.run_backfill(db=db, encoder=FakeEncoder(), settings=settings, qdrant=qdrant,
                                 image_fetcher=MockFetcher(), collection_name=name,
                                 batch_size=8, checkpoint=ckpt2)
    assert real_stats.indexed == 30
    assert qdrant.count(name).count == 30
    assert db.query("SELECT COUNT(*) c FROM embedding_map")[0]["c"] == 30


@requires_qdrant
def test_mid_batch_failure_is_retried_on_resume(tmp_path):
    """Review #3: a failed image in the middle of a batch must not be skipped
    permanently — the checkpoint must not advance past it, so resume retries."""
    db = FakeD1Client()
    _seed_images(db, 6)
    settings = _settings()
    qdrant = _new_qdrant()
    name = settings.active_collection
    ckpt_path = str(tmp_path / "ckpt.json")

    fetcher = MockFetcher()
    # img0002 fails on the first run only; img0003 (after it) succeeds.
    fail_state = {"blocked": {"img0002"}}

    def flaky_fetch(url):
        image_id = url.split("imgid=")[1].split("&")[0]
        if image_id in fail_state["blocked"]:
            raise bi.TransientFetchError(f"transient {image_id}")
        return fetcher(url)

    ckpt = bi.Checkpoint(path=ckpt_path).load()
    stats = bi.run_backfill(db=db, encoder=FakeEncoder(), settings=settings, qdrant=qdrant,
                            image_fetcher=flaky_fetch, collection_name=name,
                            batch_size=6, checkpoint=ckpt, max_retries=1, sleep=lambda s: None)
    # img0002 failed; the other 5 indexed. Checkpoint must NOT have passed img0002.
    assert stats.fetch_failures == 1
    assert stats.indexed == 5
    assert db.query("SELECT COUNT(*) c FROM embedding_map WHERE image_id='img0002'")[0]["c"] == 0
    assert ckpt.last_image_id == "img0001"  # last contiguous success before the failure

    # Resume with the fetcher now healthy: img0002 (and anything after) reattempted.
    fail_state["blocked"] = set()
    ckpt2 = bi.Checkpoint(path=ckpt_path).load()
    assert ckpt2.last_image_id == "img0001"
    stats2 = bi.run_backfill(db=db, encoder=FakeEncoder(), settings=settings, qdrant=qdrant,
                             image_fetcher=flaky_fetch, collection_name=name,
                             batch_size=6, checkpoint=ckpt2, max_retries=1, sleep=lambda s: None)
    # img0002 re-attempted and now indexed; img0003-5 skipped as already mapped.
    assert db.query("SELECT COUNT(*) c FROM embedding_map WHERE image_id='img0002'")[0]["c"] == 1
    assert db.query("SELECT COUNT(*) c FROM embedding_map")[0]["c"] == 6
    assert qdrant.count(name).count == 6
    assert db.query("SELECT COUNT(DISTINCT vector_id) c FROM embedding_map")[0]["c"] == 6
