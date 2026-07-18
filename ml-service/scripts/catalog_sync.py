"""Catalog + ImageKit inventory sync into D1 (plan Task 3).

Runs in the indexing/admin environment and writes ``products`` /
``product_images`` through :class:`~app.d1_client.D1Client`.

Design
------
* **Configurable source adapter.** The source yields rows carrying the required
  fields ``product_id``, ``image_id``, ``imagekit_file_id``, ``imagekit_url``,
  ``source_updated_at``, ``is_reference`` (plus optional product metadata:
  ``name``, ``sku``, ``manufacturer``, ``strength``, ``barcode``). The default /
  preferred adapter reads an external catalog/commerce export (CSV or JSON) that
  carries ``product_id`` per image. An ImageKit-listing adapter is a secondary
  option only when the product↔image mapping is encoded in ImageKit itself.
* **Unmapped images** (no resolvable ``product_id``) are logged to an
  ``unmapped_images`` report and skipped — never guessed. The run reports mapped
  / unmapped counts and does not fail wholesale on a few unmapped rows.
* **Soft-delete first.** Incremental sync upserts changed rows by
  ``source_updated_at``; products removed from the source are marked
  ``active=0`` and their images get ``product_images.deleted_at`` set. Rows are
  never hard-deleted here — ``reconcile.py`` clears vectors first.
* **Idempotent.** A second identical run performs zero writes.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional

from app.config import get_settings
from app.d1_client import D1Client

logger = logging.getLogger("catalog_sync")

REQUIRED_IMAGE_FIELDS = (
    "product_id",
    "image_id",
    "imagekit_file_id",
    "imagekit_url",
    "source_updated_at",
    "is_reference",
)
PRODUCT_META_FIELDS = ("sku", "name", "manufacturer", "strength", "barcode")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _to_bool_int(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0
    if value is None:
        return 0
    return 1 if str(value).strip().lower() in {"1", "true", "yes", "y", "t"} else 0


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------
def csv_source_adapter(path: str) -> Iterator[dict[str, Any]]:
    """Yield raw dict rows from a CSV export (one row per catalog image)."""
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            yield dict(row)


def json_source_adapter(path: str) -> Iterator[dict[str, Any]]:
    """Yield raw dict rows from a JSON export.

    Accepts either a top-level list of row objects or ``{"rows": [...]}``.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    for row in rows:
        yield dict(row)


def imagekit_listing_adapter(
    listing: Iterable[dict[str, Any]],
    mapping_resolver,
) -> Iterator[dict[str, Any]]:
    """Secondary adapter: map an ImageKit file listing to source rows.

    ``mapping_resolver(file) -> product_id | None`` encodes the product↔image
    relationship (folder path / filename convention / custom metadata). Files
    that resolve to ``None`` are emitted with an empty ``product_id`` so the
    sync reports them as unmapped rather than guessing.
    """
    for file in listing:
        product_id = mapping_resolver(file) or ""
        yield {
            "product_id": product_id,
            "image_id": file.get("fileId") or file.get("image_id") or "",
            "imagekit_file_id": file.get("fileId") or "",
            "imagekit_url": file.get("url") or "",
            "source_updated_at": file.get("updatedAt") or file.get("source_updated_at") or "",
            "is_reference": file.get("is_reference", False),
        }


def get_source_adapter(path: str, fmt: Optional[str] = None) -> Iterator[dict[str, Any]]:
    """Return the default catalog-export adapter iterator for ``path``.

    Format is inferred from the extension unless ``fmt`` ('csv'|'json') is given.
    """
    resolved = (fmt or path.rsplit(".", 1)[-1]).lower()
    if resolved == "csv":
        return csv_source_adapter(path)
    if resolved == "json":
        return json_source_adapter(path)
    raise ValueError(f"Unsupported source format: {resolved!r} (use csv or json)")


# ---------------------------------------------------------------------------
# Normalized row + report
# ---------------------------------------------------------------------------
@dataclass
class SourceImage:
    product_id: str
    image_id: str
    imagekit_file_id: str
    imagekit_url: str
    source_updated_at: str
    is_reference: int
    # optional product metadata
    sku: Optional[str] = None
    name: Optional[str] = None
    manufacturer: Optional[str] = None
    strength: Optional[str] = None
    barcode: Optional[str] = None


@dataclass
class SyncReport:
    mapped: int = 0
    unmapped: int = 0
    products_inserted: int = 0
    products_updated: int = 0
    images_inserted: int = 0
    images_updated: int = 0
    products_soft_deleted: int = 0
    images_soft_deleted: int = 0
    unmapped_images: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return (
            self.products_inserted
            + self.products_updated
            + self.images_inserted
            + self.images_updated
            + self.products_soft_deleted
            + self.images_soft_deleted
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mapped": self.mapped,
            "unmapped": self.unmapped,
            "products_inserted": self.products_inserted,
            "products_updated": self.products_updated,
            "images_inserted": self.images_inserted,
            "images_updated": self.images_updated,
            "products_soft_deleted": self.products_soft_deleted,
            "images_soft_deleted": self.images_soft_deleted,
            "total_changes": self.total_changes,
        }


def normalize_rows(
    raw_rows: Iterable[dict[str, Any]],
    report: SyncReport,
) -> list[SourceImage]:
    """Validate + normalize raw rows; route unmapped rows into the report."""
    images: list[SourceImage] = []
    for raw in raw_rows:
        product_id = str(raw.get("product_id") or "").strip()
        image_id = str(raw.get("image_id") or "").strip()
        # A row is "unmapped" when it lacks the product↔image link.
        if not product_id or not image_id:
            report.unmapped += 1
            report.unmapped_images.append(dict(raw))
            logger.warning(
                "unmapped image skipped (no product_id/image_id): %s",
                raw.get("imagekit_file_id") or raw.get("image_id") or raw,
            )
            continue
        report.mapped += 1
        images.append(
            SourceImage(
                product_id=product_id,
                image_id=image_id,
                imagekit_file_id=str(raw.get("imagekit_file_id") or ""),
                imagekit_url=str(raw.get("imagekit_url") or ""),
                source_updated_at=str(raw.get("source_updated_at") or ""),
                is_reference=_to_bool_int(raw.get("is_reference")),
                sku=_opt(raw.get("sku")),
                name=_opt(raw.get("name")),
                manufacturer=_opt(raw.get("manufacturer")),
                strength=_opt(raw.get("strength")),
                barcode=_opt(raw.get("barcode")),
            )
        )
    return images


def _opt(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# ---------------------------------------------------------------------------
# Sync core
# ---------------------------------------------------------------------------
def _fetch_existing_products(db: D1Client) -> dict[str, dict[str, Any]]:
    rows = db.query("SELECT product_id, sku, name, manufacturer, strength, barcode, active FROM products")
    return {r["product_id"]: r for r in rows}


def _fetch_existing_images(db: D1Client) -> dict[str, dict[str, Any]]:
    rows = db.query(
        "SELECT image_id, product_id, imagekit_file_id, imagekit_url, "
        "is_reference, source_updated_at, deleted_at FROM product_images"
    )
    return {r["image_id"]: r for r in rows}


def _product_from_images(images: list[SourceImage]) -> dict[str, dict[str, Any]]:
    """Collapse image rows into one product-metadata record per product_id."""
    products: dict[str, dict[str, Any]] = {}
    for img in images:
        prod = products.setdefault(
            img.product_id,
            {"product_id": img.product_id, "sku": None, "name": None,
             "manufacturer": None, "strength": None, "barcode": None},
        )
        for f in PRODUCT_META_FIELDS:
            val = getattr(img, f)
            if val is not None and prod.get(f) is None:
                prod[f] = val
    return products


def sync(
    db: D1Client,
    raw_rows: Iterable[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> SyncReport:
    """Sync source rows into D1. Returns a :class:`SyncReport`.

    When ``dry_run`` is True the plan is computed and reported but no writes are
    issued (change counts still reflect what *would* change).
    """
    report = SyncReport()
    images = normalize_rows(raw_rows, report)

    existing_products = _fetch_existing_products(db)
    existing_images = _fetch_existing_images(db)
    source_products = _product_from_images(images)

    statements: list[tuple[str, list[Any]]] = []
    now = _now_iso()

    # --- products upsert -------------------------------------------------
    for pid, meta in source_products.items():
        name = meta.get("name") or pid  # products.name is NOT NULL
        existing = existing_products.get(pid)
        if existing is None:
            report.products_inserted += 1
            statements.append((
                "INSERT INTO products (product_id, sku, name, manufacturer, strength, barcode, active, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                [pid, meta.get("sku"), name, meta.get("manufacturer"),
                 meta.get("strength"), meta.get("barcode"), now],
            ))
        else:
            changed = (
                (existing.get("sku") or None) != meta.get("sku")
                or (existing.get("name") or None) != name
                or (existing.get("manufacturer") or None) != meta.get("manufacturer")
                or (existing.get("strength") or None) != meta.get("strength")
                or (existing.get("barcode") or None) != meta.get("barcode")
                or int(existing.get("active") or 0) != 1  # reactivate if returned
            )
            if changed:
                report.products_updated += 1
                statements.append((
                    "UPDATE products SET sku = ?, name = ?, manufacturer = ?, strength = ?, "
                    "barcode = ?, active = 1, updated_at = ? WHERE product_id = ?",
                    [meta.get("sku"), name, meta.get("manufacturer"), meta.get("strength"),
                     meta.get("barcode"), now, pid],
                ))

    # --- images upsert (by source_updated_at + field changes) -----------
    source_image_ids = {img.image_id for img in images}
    for img in images:
        existing = existing_images.get(img.image_id)
        if existing is None:
            report.images_inserted += 1
            statements.append((
                "INSERT INTO product_images (image_id, product_id, imagekit_file_id, imagekit_url, "
                "is_reference, source_updated_at, deleted_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
                [img.image_id, img.product_id, img.imagekit_file_id, img.imagekit_url,
                 img.is_reference, img.source_updated_at, now],
            ))
        else:
            changed = (
                (existing.get("source_updated_at") or "") != img.source_updated_at
                or (existing.get("product_id") or "") != img.product_id
                or (existing.get("imagekit_file_id") or "") != img.imagekit_file_id
                or (existing.get("imagekit_url") or "") != img.imagekit_url
                or int(existing.get("is_reference") or 0) != img.is_reference
                or existing.get("deleted_at") is not None  # un-delete if returned
            )
            if changed:
                report.images_updated += 1
                statements.append((
                    "UPDATE product_images SET product_id = ?, imagekit_file_id = ?, imagekit_url = ?, "
                    "is_reference = ?, source_updated_at = ?, deleted_at = NULL WHERE image_id = ?",
                    [img.product_id, img.imagekit_file_id, img.imagekit_url,
                     img.is_reference, img.source_updated_at, img.image_id],
                ))

    # --- soft-delete products removed from source -----------------------
    for pid, existing in existing_products.items():
        if pid not in source_products and int(existing.get("active") or 0) == 1:
            report.products_soft_deleted += 1
            statements.append((
                "UPDATE products SET active = 0, updated_at = ? WHERE product_id = ?",
                [now, pid],
            ))

    # --- soft-delete images gone from source (live rows only) -----------
    for image_id, existing in existing_images.items():
        gone = image_id not in source_image_ids
        if gone and existing.get("deleted_at") is None:
            report.images_soft_deleted += 1
            statements.append((
                "UPDATE product_images SET deleted_at = ? WHERE image_id = ?",
                [now, image_id],
            ))

    if not dry_run and statements:
        db.batch(statements)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _write_unmapped_report(report: SyncReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.unmapped_images, fh, indent=2)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sync catalog/ImageKit inventory into D1.")
    parser.add_argument("--source", required=True, help="Path to CSV/JSON catalog export.")
    parser.add_argument("--format", choices=["csv", "json"], default=None, help="Override format inference.")
    parser.add_argument("--dry-run", action="store_true", help="Compute + report changes without writing.")
    parser.add_argument("--unmapped-report", default=None, help="Where to write the unmapped_images JSON report.")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=args.log_level or settings.log_level)

    db = D1Client(settings)
    rows = get_source_adapter(args.source, args.format)
    report = sync(db, rows, dry_run=args.dry_run)

    if args.unmapped_report:
        _write_unmapped_report(report, args.unmapped_report)

    logger.info("catalog_sync report: %s", json.dumps(report.as_dict()))
    print(json.dumps(report.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
