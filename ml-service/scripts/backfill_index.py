"""Idempotent, resumable catalog-image indexing (plan Task 11).

Pipeline (runs in the indexing/GPU environment):

1. Iterate ``product_images`` from D1, paginated, skipping soft-deleted rows.
2. Skip image_ids already mapped for the *current encoder* (idempotent/resume).
3. Fetch a **resized** ImageKit transform (default 512px via URL transform
   params) to cut bandwidth, with retry + exponential backoff on transient
   errors.
4. Embed in GPU batches via ``get_encoder()``.
5. Upsert deterministic-id vectors into the encoder's per-encoder Qdrant
   collection and write ``embedding_map`` rows in D1.
6. Persist a checkpoint (cursor) file so a killed run resumes without
   re-indexing or duplicating ids.
7. Emit throughput (images/sec) + a GPU-hours estimate at the end.

The heavy dependencies (encoder / Qdrant / ImageKit HTTP) are injected so tests
can substitute a deterministic fake encoder, an in-memory Qdrant, and a mock
image fetcher. Real model/GPU work is never required to exercise the logic.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

from app.config import Settings, get_settings
from app.d1_client import D1Client
from app.search import qdrant_client as qc

logger = logging.getLogger("backfill_index")

DEFAULT_TRANSFORM_WIDTH = 512
DEFAULT_BATCH = 32
DEFAULT_PAGE_SIZE = 500
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE = 0.5


# ---------------------------------------------------------------------------
# ImageKit resized-transform URL + fetch
# ---------------------------------------------------------------------------
def build_transform_url(imagekit_url: str, width: int = DEFAULT_TRANSFORM_WIDTH) -> str:
    """Return a resized ImageKit URL using the ``tr=w-<width>`` query param.

    ImageKit accepts transforms either as a path segment or the ``tr`` query
    parameter; the query form composes safely with arbitrary catalog URLs.
    """
    if not imagekit_url:
        return imagekit_url
    sep = "&" if "?" in imagekit_url else "?"
    return f"{imagekit_url}{sep}tr=w-{width}"


class TransientFetchError(RuntimeError):
    """Raised for retryable ImageKit/D1 fetch failures."""


def default_image_fetcher(url: str, timeout: float = 30.0) -> bytes:
    """Fetch image bytes over HTTP. Raises TransientFetchError on 5xx/timeout."""
    import httpx  # noqa: WPS433

    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError as exc:  # network/timeout
        raise TransientFetchError(str(exc)) from exc
    if resp.status_code >= 500:
        raise TransientFetchError(f"ImageKit HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise RuntimeError(f"ImageKit HTTP {resp.status_code} for {url}")
    return resp.content


def retry_with_backoff(
    fn: Callable[[], Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base: float = DEFAULT_BACKOFF_BASE,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call ``fn`` retrying TransientFetchError with exponential backoff."""
    attempt = 0
    while True:
        try:
            return fn()
        except TransientFetchError:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep(base * (2 ** (attempt - 1)))


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
@dataclass
class Checkpoint:
    """Resume cursor persisted to disk between/within runs."""

    path: str
    last_image_id: Optional[str] = None
    indexed: int = 0

    def load(self) -> "Checkpoint":
        if self.path and os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.last_image_id = data.get("last_image_id")
            self.indexed = int(data.get("indexed", 0))
        return self

    def save(self) -> None:
        if not self.path:
            return
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"last_image_id": self.last_image_id, "indexed": self.indexed}, fh)
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# D1 helpers
# ---------------------------------------------------------------------------
def iter_live_images(
    db: D1Client,
    *,
    after_image_id: Optional[str] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Iterator[dict[str, Any]]:
    """Yield live (not soft-deleted) product_images ordered by image_id.

    Ordering by ``image_id`` gives a stable cursor for resume. ``after_image_id``
    resumes strictly after a given id.
    """
    params: list[Any] = []
    where = "deleted_at IS NULL"
    if after_image_id is not None:
        where += " AND image_id > ?"
        params.append(after_image_id)
    sql = (
        "SELECT image_id, product_id, imagekit_file_id, imagekit_url "
        f"FROM product_images WHERE {where} ORDER BY image_id"
    )
    yield from db.paginate(sql, params, page_size=page_size)


def already_mapped_ids(db: D1Client, encoder: str) -> set[str]:
    """Return the set of image_ids already indexed for ``encoder``."""
    rows = db.query("SELECT image_id FROM embedding_map WHERE encoder = ?", [encoder])
    return {r["image_id"] for r in rows}


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------
@dataclass
class BackfillStats:
    indexed: int = 0
    skipped_existing: int = 0
    fetch_failures: int = 0
    seconds: float = 0.0
    gpu_hours_estimate: float = 0.0

    @property
    def images_per_sec(self) -> float:
        return (self.indexed / self.seconds) if self.seconds > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "indexed": self.indexed,
            "skipped_existing": self.skipped_existing,
            "fetch_failures": self.fetch_failures,
            "seconds": round(self.seconds, 3),
            "images_per_sec": round(self.images_per_sec, 3),
            "gpu_hours_estimate": round(self.gpu_hours_estimate, 6),
        }


def _load_image(data: bytes):
    import io

    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def run_backfill(
    *,
    db: D1Client,
    encoder,
    settings: Optional[Settings] = None,
    qdrant=None,
    image_fetcher: Callable[[str], bytes] = default_image_fetcher,
    collection_name: Optional[str] = None,
    transform_width: int = DEFAULT_TRANSFORM_WIDTH,
    batch_size: Optional[int] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    checkpoint: Optional[Checkpoint] = None,
    dry_run: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> BackfillStats:
    """Index all live product_images not yet mapped for the current encoder.

    Parameters are injected so tests can supply a fake encoder / in-memory
    Qdrant / mock fetcher. Returns :class:`BackfillStats` with throughput and a
    GPU-hours estimate.
    """
    settings = settings or get_settings()
    encoder_name = settings.encoder
    embedding_dim = settings.embedding_dim
    batch_size = batch_size or settings.batch_size or DEFAULT_BATCH
    collection_name = collection_name or settings.active_collection

    qdrant = qdrant or qc.get_client(settings)
    qc.ensure_collection(settings, client=qdrant, collection_name=collection_name, embedding_dim=embedding_dim)

    mapped = already_mapped_ids(db, encoder_name)
    stats = BackfillStats()
    started = time.perf_counter()

    resume_after = checkpoint.last_image_id if checkpoint else None

    # Buffer rows into GPU batches.
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        images = []
        kept: list[dict[str, Any]] = []
        for row in pending:
            url = build_transform_url(row.get("imagekit_url") or "", transform_width)
            try:
                data = retry_with_backoff(
                    lambda u=url: image_fetcher(u), max_retries=max_retries, sleep=sleep
                )
            except TransientFetchError:
                stats.fetch_failures += 1
                logger.warning("giving up on image after retries: %s", row.get("image_id"))
                continue
            images.append(_load_image(data))
            kept.append(row)
        if not kept:
            pending.clear()
            return

        vectors = encoder.embed(images)
        points = []
        map_stmts: list[tuple[str, list[Any]]] = []
        for row, vec in zip(kept, vectors):
            image_id = row["image_id"]
            product_id = row["product_id"]
            vid = qc.deterministic_vector_id(encoder_name, embedding_dim, image_id)
            points.append(
                qc.build_point(
                    vec,
                    image_id=image_id,
                    product_id=product_id,
                    encoder=encoder_name,
                    embedding_dim=embedding_dim,
                    is_reference=bool(row.get("is_reference", False)),
                    vector_id=vid,
                )
            )
            map_stmts.append((
                "INSERT OR REPLACE INTO embedding_map "
                "(vector_id, image_id, product_id, encoder, embedding_dim, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                [vid, image_id, product_id, encoder_name, embedding_dim],
            ))

        if not dry_run:
            qc.upsert(points, settings=settings, client=qdrant, collection_name=collection_name)
            db.batch(map_stmts)

        stats.indexed += len(kept)
        mapped.update(row["image_id"] for row in kept)
        if checkpoint:
            checkpoint.last_image_id = kept[-1]["image_id"]
            checkpoint.indexed = stats.indexed
            checkpoint.save()
        pending.clear()

    for row in iter_live_images(db, after_image_id=resume_after, page_size=page_size):
        if row["image_id"] in mapped:
            stats.skipped_existing += 1
            continue
        pending.append(row)
        if len(pending) >= batch_size:
            flush()
    flush()

    stats.seconds = time.perf_counter() - started
    # Rough GPU-hours estimate: wall-clock seconds spent / 3600. On real GPU
    # runs this tracks embed throughput; with a fake encoder it is ~0.
    stats.gpu_hours_estimate = stats.seconds / 3600.0
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill/index catalog images into Qdrant.")
    parser.add_argument("--checkpoint", default=".backfill_checkpoint.json")
    parser.add_argument("--transform-width", type=int, default=DEFAULT_TRANSFORM_WIDTH)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=args.log_level or settings.log_level)

    from app.encoders import get_encoder

    db = D1Client(settings)
    encoder = get_encoder(settings)
    checkpoint = Checkpoint(path=args.checkpoint).load()

    stats = run_backfill(
        db=db,
        encoder=encoder,
        settings=settings,
        transform_width=args.transform_width,
        batch_size=args.batch_size,
        page_size=args.page_size,
        checkpoint=checkpoint,
        dry_run=args.dry_run,
    )
    logger.info("backfill stats: %s", json.dumps(stats.as_dict()))
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
