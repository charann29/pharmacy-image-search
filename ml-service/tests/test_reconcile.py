"""Tests for scripts/reconcile.py.

Uses the SQLite-backed fake D1 + in-memory Qdrant. Injects:
  * an orphan  — a Qdrant point with no embedding_map row;
  * a missing  — an embedding_map row with no Qdrant point;
  * a deactivated product + a soft-deleted image whose vectors must be cleared.

Asserts reconcile repairs to a consistent state and reports correct counts.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.search import qdrant_client as qc

from _fake_d1 import FakeD1Client
from conftest import requires_qdrant

from scripts import reconcile as rec

DIM = 8


def _settings():
    return Settings(encoder="siglip2", embedding_dim_override=DIM,
                    validate_collection_on_startup=False)


def _new_qdrant():
    from qdrant_client import QdrantClient

    return QdrantClient(":memory:")


def _vec(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _add_product(db, pid, active=1):
    db.execute("INSERT INTO products (product_id, name, active) VALUES (?, ?, ?)",
               [pid, f"name {pid}", active])


def _add_image(db, iid, pid, deleted=False):
    db.execute(
        "INSERT INTO product_images (image_id, product_id, imagekit_file_id, imagekit_url, "
        "is_reference, source_updated_at, deleted_at, created_at) "
        "VALUES (?, ?, ?, ?, 0, '2026-01-01T00:00:00Z', ?, '2026-01-01T00:00:00Z')",
        [iid, pid, f"f-{iid}", f"https://ik/{iid}", "2026-03-01T00:00:00Z" if deleted else None],
    )


def _index(db, qdrant, name, settings, iid, pid, seed, write_map=True, write_vec=True):
    vid = qc.deterministic_vector_id("siglip2", DIM, iid)
    if write_vec:
        point = qc.build_point(_vec(seed), image_id=iid, product_id=pid,
                               encoder="siglip2", embedding_dim=DIM, vector_id=vid)
        qc.upsert([point], settings=settings, client=qdrant, collection_name=name)
    if write_map:
        db.execute(
            "INSERT INTO embedding_map (vector_id, image_id, product_id, encoder, embedding_dim) "
            "VALUES (?, ?, ?, 'siglip2', ?)",
            [vid, iid, pid, DIM],
        )
    return vid


@requires_qdrant
def test_reconcile_repairs_orphan_and_missing():
    db = FakeD1Client()
    settings = _settings()
    qdrant = _new_qdrant()
    name = qc.ensure_collection(settings, client=qdrant, collection_name=settings.active_collection)

    _add_product(db, "p1")
    _add_image(db, "i1", "p1")
    _add_image(db, "i2", "p1")

    # consistent row
    _index(db, qdrant, name, settings, "i1", "p1", seed=1)
    # orphan: vector present, no map row
    _index(db, qdrant, name, settings, "i2", "p1", seed=2, write_map=False)
    # missing: map row present, no vector
    _add_image(db, "i3", "p1")
    _index(db, qdrant, name, settings, "i3", "p1", seed=3, write_vec=False)

    report = rec.reconcile(db=db, settings=settings, qdrant=qdrant, collection_name=name)
    assert report.orphans_deleted == 1
    assert report.missing_repaired == 1

    # Consistent state: qdrant ids == embedding_map vector ids.
    q_ids = rec.scroll_vector_ids(qdrant, name)
    m_ids = {r["vector_id"] for r in db.query("SELECT vector_id FROM embedding_map")}
    assert q_ids == m_ids
    # Only i1 survives.
    assert len(q_ids) == 1


@requires_qdrant
def test_reconcile_deletes_deactivated_and_soft_deleted():
    db = FakeD1Client()
    settings = _settings()
    qdrant = _new_qdrant()
    name = qc.ensure_collection(settings, client=qdrant, collection_name=settings.active_collection)

    _add_product(db, "p1", active=1)
    _add_product(db, "p2", active=0)  # deactivated product
    _add_image(db, "i1", "p1")
    _add_image(db, "i2", "p2")            # belongs to deactivated product
    _add_image(db, "i3", "p1", deleted=True)  # soft-deleted image

    _index(db, qdrant, name, settings, "i1", "p1", seed=1)
    _index(db, qdrant, name, settings, "i2", "p2", seed=2)
    _index(db, qdrant, name, settings, "i3", "p1", seed=3)

    report = rec.reconcile(db=db, settings=settings, qdrant=qdrant, collection_name=name)
    assert report.deactivated_vectors_deleted == 2  # i2 (inactive product) + i3 (soft-deleted)

    q_ids = rec.scroll_vector_ids(qdrant, name)
    assert len(q_ids) == 1  # only i1 remains
    remaining = db.query("SELECT image_id FROM embedding_map")
    assert [r["image_id"] for r in remaining] == ["i1"]


@requires_qdrant
def test_reconcile_hard_delete_after_vectors_cleared():
    db = FakeD1Client()
    settings = _settings()
    qdrant = _new_qdrant()
    name = qc.ensure_collection(settings, client=qdrant, collection_name=settings.active_collection)

    _add_product(db, "p1")
    _add_image(db, "i1", "p1")
    _add_image(db, "i3", "p1", deleted=True)
    _index(db, qdrant, name, settings, "i1", "p1", seed=1)
    _index(db, qdrant, name, settings, "i3", "p1", seed=3)

    report = rec.reconcile(db=db, settings=settings, qdrant=qdrant, collection_name=name,
                           hard_delete_images=True)
    assert report.deactivated_vectors_deleted == 1
    assert report.images_hard_deleted == 1
    # i3 row hard-deleted now its vector + mapping are gone.
    assert db.query("SELECT COUNT(*) c FROM product_images WHERE image_id='i3'")[0]["c"] == 0
    # i1 (live) untouched.
    assert db.query("SELECT COUNT(*) c FROM product_images WHERE image_id='i1'")[0]["c"] == 1


@requires_qdrant
def test_reconcile_dry_run_changes_nothing():
    db = FakeD1Client()
    settings = _settings()
    qdrant = _new_qdrant()
    name = qc.ensure_collection(settings, client=qdrant, collection_name=settings.active_collection)

    _add_product(db, "p1")
    _add_image(db, "i1", "p1")
    _add_image(db, "i2", "p1")
    _index(db, qdrant, name, settings, "i1", "p1", seed=1)
    _index(db, qdrant, name, settings, "i2", "p1", seed=2, write_map=False)  # orphan

    report = rec.reconcile(db=db, settings=settings, qdrant=qdrant, collection_name=name, dry_run=True)
    assert report.orphans_deleted == 1  # reported
    assert qdrant.count(name).count == 2  # but nothing deleted


@requires_qdrant
def test_reconcile_clean_state_no_changes():
    db = FakeD1Client()
    settings = _settings()
    qdrant = _new_qdrant()
    name = qc.ensure_collection(settings, client=qdrant, collection_name=settings.active_collection)

    _add_product(db, "p1")
    _add_image(db, "i1", "p1")
    _index(db, qdrant, name, settings, "i1", "p1", seed=1)

    report = rec.reconcile(db=db, settings=settings, qdrant=qdrant, collection_name=name)
    assert report.as_dict() == {
        "orphans_deleted": 0, "missing_repaired": 0,
        "deactivated_vectors_deleted": 0, "images_hard_deleted": 0,
    }
