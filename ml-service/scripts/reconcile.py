"""Qdrant <-> D1 consistency repair (plan Task 11).

Repairs three classes of drift for the current encoder's collection:

1. **Orphan vectors** — Qdrant points with no matching ``embedding_map`` row in
   D1. These are deleted from Qdrant.
2. **Missing vectors** — ``embedding_map`` rows whose vector is absent from
   Qdrant. The stale mapping row is removed so ``backfill_index`` re-indexes the
   image on its next run (reconcile does not re-embed — no encoder/GPU needed).
3. **Deactivated / soft-deleted** — vectors whose product is ``active=0`` or
   whose image row is soft-deleted (``deleted_at`` set). Their vectors are
   deleted from Qdrant and the ``embedding_map`` rows removed; the now-unmapped
   soft-deleted image rows may then be hard-deleted (optional, ``--hard-delete``).

Reports counts of orphans / missing / deactivated repaired.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, Optional

from app.config import Settings, get_settings
from app.d1_client import D1Client
from app.search import qdrant_client as qc

logger = logging.getLogger("reconcile")

SCROLL_PAGE = 1000


@dataclass
class ReconcileReport:
    orphans_deleted: int = 0
    missing_repaired: int = 0
    deactivated_vectors_deleted: int = 0
    images_hard_deleted: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "orphans_deleted": self.orphans_deleted,
            "missing_repaired": self.missing_repaired,
            "deactivated_vectors_deleted": self.deactivated_vectors_deleted,
            "images_hard_deleted": self.images_hard_deleted,
        }


def scroll_vector_ids(qdrant, collection_name: str, page: int = SCROLL_PAGE) -> set[str]:
    """Return the full set of point ids currently in the Qdrant collection."""
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection_name,
            with_payload=False,
            with_vectors=False,
            limit=page,
            offset=offset,
        )
        for p in points:
            ids.add(str(p.id))
        if offset is None:
            break
    return ids


def _map_rows_for_encoder(db: D1Client, encoder: str) -> list[dict[str, Any]]:
    return db.query(
        "SELECT vector_id, image_id, product_id FROM embedding_map WHERE encoder = ?",
        [encoder],
    )


def _deactivated_vector_ids(db: D1Client, encoder: str) -> list[dict[str, Any]]:
    """embedding_map rows whose product is inactive or image is soft-deleted."""
    return db.query(
        "SELECT em.vector_id, em.image_id FROM embedding_map em "
        "LEFT JOIN products p ON p.product_id = em.product_id "
        "LEFT JOIN product_images pi ON pi.image_id = em.image_id "
        "WHERE em.encoder = ? AND ("
        "  p.product_id IS NULL OR p.active = 0 "
        "  OR pi.image_id IS NULL OR pi.deleted_at IS NOT NULL"
        ")",
        [encoder],
    )


def reconcile(
    *,
    db: D1Client,
    settings: Optional[Settings] = None,
    qdrant=None,
    collection_name: Optional[str] = None,
    hard_delete_images: bool = False,
    dry_run: bool = False,
) -> ReconcileReport:
    """Repair Qdrant<->D1 drift for the current encoder. Returns a report."""
    settings = settings or get_settings()
    encoder = settings.encoder
    collection_name = collection_name or settings.active_collection
    qdrant = qdrant or qc.get_client(settings)

    report = ReconcileReport()

    qdrant_ids = scroll_vector_ids(qdrant, collection_name)
    map_rows = _map_rows_for_encoder(db, encoder)
    map_by_vector = {r["vector_id"]: r for r in map_rows}
    map_ids = set(map_by_vector)

    # --- 1. deactivated / soft-deleted: remove vectors + mapping rows ----
    deactivated = _deactivated_vector_ids(db, encoder)
    deactivated_vids = [r["vector_id"] for r in deactivated]
    deactivated_set = set(deactivated_vids)
    present_deactivated = [v for v in deactivated_vids if v in qdrant_ids]
    if deactivated_vids:
        if not dry_run:
            if present_deactivated:
                qdrant.delete(collection_name=collection_name, points_selector=present_deactivated)
            db.batch([
                ("DELETE FROM embedding_map WHERE vector_id = ?", [v]) for v in deactivated_vids
            ])
        report.deactivated_vectors_deleted += len(deactivated_vids)
        # Those vectors/rows are no longer live for the checks below.
        qdrant_ids -= set(present_deactivated)
        map_ids -= deactivated_set
        for v in deactivated_set:
            map_by_vector.pop(v, None)

    # --- 2. orphan vectors: in Qdrant, no D1 mapping -> delete -----------
    orphans = sorted(qdrant_ids - map_ids)
    if orphans:
        if not dry_run:
            qdrant.delete(collection_name=collection_name, points_selector=orphans)
        report.orphans_deleted += len(orphans)

    # --- 3. missing vectors: D1 mapping row, no Qdrant point -> drop row --
    missing = sorted(map_ids - qdrant_ids)
    if missing:
        if not dry_run:
            db.batch([("DELETE FROM embedding_map WHERE vector_id = ?", [v]) for v in missing])
        report.missing_repaired += len(missing)

    # --- optional: hard-delete now-unmapped soft-deleted image rows ------
    if hard_delete_images:
        removable = db.query(
            "SELECT pi.image_id FROM product_images pi "
            "LEFT JOIN embedding_map em ON em.image_id = pi.image_id "
            "WHERE pi.deleted_at IS NOT NULL AND em.image_id IS NULL"
        )
        image_ids = [r["image_id"] for r in removable]
        if image_ids:
            if not dry_run:
                db.batch([
                    ("DELETE FROM product_images WHERE image_id = ?", [i]) for i in image_ids
                ])
            report.images_hard_deleted += len(image_ids)

    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Repair Qdrant<->D1 consistency.")
    parser.add_argument("--hard-delete", action="store_true",
                        help="Hard-delete soft-deleted image rows once their vectors are cleared.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=args.log_level or settings.log_level)

    db = D1Client(settings)
    report = reconcile(
        db=db,
        settings=settings,
        hard_delete_images=args.hard_delete,
        dry_run=args.dry_run,
    )
    logger.info("reconcile report: %s", json.dumps(report.as_dict()))
    print(json.dumps(report.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
