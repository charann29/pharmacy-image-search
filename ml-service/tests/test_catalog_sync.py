"""Tests for scripts/catalog_sync.py.

Uses a SQLite-backed fake of the D1 client (``0001_init.sql`` applied) so the
real sync logic runs end to end without network calls. Covers: dry-run counts,
incremental change + soft-delete, unmapped-image reporting/skip, idempotent
re-run (0 changes).
"""
from __future__ import annotations

import csv
import json

import pytest

from _fake_d1 import FakeD1Client
from scripts import catalog_sync as cs


def _rows():
    return [
        {"product_id": "p1", "image_id": "i1", "imagekit_file_id": "f1",
         "imagekit_url": "https://ik/i1", "source_updated_at": "2026-01-01T00:00:00Z",
         "is_reference": "1", "name": "Crocin 650"},
        {"product_id": "p1", "image_id": "i2", "imagekit_file_id": "f2",
         "imagekit_url": "https://ik/i2", "source_updated_at": "2026-01-01T00:00:00Z",
         "is_reference": "0", "name": "Crocin 650"},
        {"product_id": "p2", "image_id": "i3", "imagekit_file_id": "f3",
         "imagekit_url": "https://ik/i3", "source_updated_at": "2026-01-01T00:00:00Z",
         "is_reference": "1", "name": "Dolo 650"},
    ]


def test_dry_run_reports_but_does_not_write():
    db = FakeD1Client()
    report = cs.sync(db, _rows(), dry_run=True)
    assert report.mapped == 3
    assert report.products_inserted == 2
    assert report.images_inserted == 3
    # Nothing actually written.
    assert db.query("SELECT COUNT(*) c FROM products")[0]["c"] == 0
    assert db.query("SELECT COUNT(*) c FROM product_images")[0]["c"] == 0


def test_full_sync_populates_correct_counts():
    db = FakeD1Client()
    report = cs.sync(db, _rows())
    assert report.products_inserted == 2
    assert report.images_inserted == 3
    assert db.query("SELECT COUNT(*) c FROM products")[0]["c"] == 2
    assert db.query("SELECT COUNT(*) c FROM product_images")[0]["c"] == 3
    # is_reference persisted correctly.
    img = db.query("SELECT is_reference FROM product_images WHERE image_id = 'i1'")[0]
    assert img["is_reference"] == 1


def test_idempotent_second_run_zero_changes():
    db = FakeD1Client()
    cs.sync(db, _rows())
    report2 = cs.sync(db, _rows())
    assert report2.total_changes == 0


def test_incremental_change_and_soft_delete():
    db = FakeD1Client()
    cs.sync(db, _rows())

    # One changed image (i2 updated_at bumped) and product p2 removed entirely.
    changed = [
        {"product_id": "p1", "image_id": "i1", "imagekit_file_id": "f1",
         "imagekit_url": "https://ik/i1", "source_updated_at": "2026-01-01T00:00:00Z",
         "is_reference": "1", "name": "Crocin 650"},
        {"product_id": "p1", "image_id": "i2", "imagekit_file_id": "f2",
         "imagekit_url": "https://ik/i2-new", "source_updated_at": "2026-02-01T00:00:00Z",
         "is_reference": "0", "name": "Crocin 650"},
    ]
    report = cs.sync(db, changed)
    assert report.images_updated == 1
    assert report.products_soft_deleted == 1  # p2 gone
    assert report.images_soft_deleted == 1   # i3 gone

    p2 = db.query("SELECT active FROM products WHERE product_id = 'p2'")[0]
    assert p2["active"] == 0
    i3 = db.query("SELECT deleted_at FROM product_images WHERE image_id = 'i3'")[0]
    assert i3["deleted_at"] is not None
    # i3 row still present (soft delete, not hard delete).
    assert db.query("SELECT COUNT(*) c FROM product_images")[0]["c"] == 3
    i2 = db.query("SELECT imagekit_url, source_updated_at FROM product_images WHERE image_id = 'i2'")[0]
    assert i2["imagekit_url"] == "https://ik/i2-new"


def test_unmapped_image_reported_and_skipped():
    db = FakeD1Client()
    rows = _rows() + [
        {"product_id": "", "image_id": "orphan1", "imagekit_file_id": "fx",
         "imagekit_url": "https://ik/orphan", "source_updated_at": "2026-01-01T00:00:00Z",
         "is_reference": "0"},
    ]
    report = cs.sync(db, rows)
    assert report.unmapped == 1
    assert report.mapped == 3
    assert len(report.unmapped_images) == 1
    # Not inserted.
    assert db.query("SELECT COUNT(*) c FROM product_images WHERE image_id = 'orphan1'")[0]["c"] == 0


def test_missing_imagekit_url_reported_and_skipped():
    db = FakeD1Client()
    rows = _rows() + [
        {"product_id": "p9", "image_id": "bad_url", "imagekit_file_id": "fx",
         "imagekit_url": "", "source_updated_at": "2026-01-01T00:00:00Z",
         "is_reference": "0", "name": "Bad"},
    ]
    report = cs.sync(db, rows)
    assert report.invalid == 1
    assert report.mapped == 3
    # Reported as skipped, never inserted (would poison D1 + break backfill).
    assert db.query("SELECT COUNT(*) c FROM product_images WHERE image_id = 'bad_url'")[0]["c"] == 0
    # Its product is not created from an invalid-only row.
    assert db.query("SELECT COUNT(*) c FROM products WHERE product_id = 'p9'")[0]["c"] == 0


def test_missing_source_updated_at_reported_and_skipped():
    db = FakeD1Client()
    rows = _rows() + [
        {"product_id": "p9", "image_id": "bad_ts", "imagekit_file_id": "fx",
         "imagekit_url": "https://ik/bad_ts", "source_updated_at": "",
         "is_reference": "0", "name": "Bad"},
    ]
    report = cs.sync(db, rows)
    assert report.invalid == 1
    assert report.mapped == 3
    assert db.query("SELECT COUNT(*) c FROM product_images WHERE image_id = 'bad_ts'")[0]["c"] == 0


def test_missing_imagekit_file_id_reported_and_skipped():
    db = FakeD1Client()
    rows = _rows() + [
        {"product_id": "p9", "image_id": "bad_fid", "imagekit_file_id": "",
         "imagekit_url": "https://ik/bad_fid", "source_updated_at": "2026-01-01T00:00:00Z",
         "is_reference": "0", "name": "Bad"},
    ]
    report = cs.sync(db, rows)
    assert report.invalid == 1
    assert db.query("SELECT COUNT(*) c FROM product_images WHERE image_id = 'bad_fid'")[0]["c"] == 0


def test_reactivation_of_previously_deleted(tmp_path):
    db = FakeD1Client()
    cs.sync(db, _rows())
    # Remove p2 -> soft delete.
    cs.sync(db, _rows()[:2])
    assert db.query("SELECT active FROM products WHERE product_id='p2'")[0]["active"] == 0
    # p2 returns -> reactivated + image un-deleted.
    report = cs.sync(db, _rows())
    assert report.products_updated == 1
    assert report.images_updated == 1
    assert db.query("SELECT active FROM products WHERE product_id='p2'")[0]["active"] == 1
    assert db.query("SELECT deleted_at FROM product_images WHERE image_id='i3'")[0]["deleted_at"] is None


# --------------------------------------------------------------------------
# CSV / JSON source adapters
# --------------------------------------------------------------------------
def test_csv_adapter_and_end_to_end(tmp_path):
    path = tmp_path / "catalog.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["product_id", "image_id", "imagekit_file_id", "imagekit_url",
                        "source_updated_at", "is_reference", "name"],
        )
        w.writeheader()
        for r in _rows():
            w.writerow(r)
    db = FakeD1Client()
    rows = cs.get_source_adapter(str(path))
    report = cs.sync(db, rows)
    assert report.images_inserted == 3


def test_json_adapter(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"rows": _rows()}))
    rows = list(cs.get_source_adapter(str(path)))
    assert len(rows) == 3
    assert rows[0]["product_id"] == "p1"


def test_imagekit_listing_adapter_marks_unmapped():
    listing = [
        {"fileId": "f1", "url": "https://ik/f1", "updatedAt": "2026-01-01T00:00:00Z"},
        {"fileId": "f2", "url": "https://ik/f2", "updatedAt": "2026-01-01T00:00:00Z"},
    ]

    def resolver(f):
        return "p1" if f["fileId"] == "f1" else None

    out = list(cs.imagekit_listing_adapter(listing, resolver))
    assert out[0]["product_id"] == "p1"
    assert out[1]["product_id"] == ""  # unmapped -> reported by sync, not guessed
