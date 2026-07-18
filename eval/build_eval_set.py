"""Assemble / generate a labeled photo->product eval set (Task 10).

The eval set is two JSONL manifests under a dataset directory:

``catalog.jsonl`` — one line per *reference* catalog image to index::

    {"product_id": "P0001", "image_id": "P0001-ref-0",
     "image_path": "images/P0001-ref-0.png", "is_reference": true,
     "category": "analgesic", "source_updated_at": "2026-07-01T00:00:00Z"}

``queries.jsonl`` — one line per labeled *query photo* (the photo a user
snaps)::

    {"query_image_path": "images/P0001-q0.png", "expected_product_id": "P0001",
     "category": "analgesic", "in_catalog": true}

Fields:
  * ``query_image_path`` / ``image_path`` — path relative to the dataset dir.
  * ``expected_product_id`` — the correct product; ``null`` for an
    out-of-catalog negative (``in_catalog: false``) used for false-positive
    / weak-match calibration.
  * ``category`` — optional grouping label (analgesic, antibiotic, ...).
  * ``is_reference`` — reference image flag (Task 2 payload).
  * ``source_updated_at`` — optional catalog freshness timestamp; ``run_eval``
    can compare it against index freshness.

This module has two entry points:

  * :func:`generate_synthetic_dataset` — build a small, fully synthetic sample
    fixture (deterministic PNGs + both manifests + a README). Used to ship the
    committed ``eval/dataset`` fixture so ``run_eval.py`` runs end-to-end in CI
    with no real catalog data or model downloads.
  * :func:`assemble_from_rows` — assemble manifests from an external
    catalog/commerce export (rows already carrying ``product_id`` per image);
    the preferred production path (mirrors ``catalog_sync`` adapter contract).

CLI::

    python eval/build_eval_set.py --out eval/dataset --synthetic
    python eval/build_eval_set.py --out eval/dataset --from-csv export.csv
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

CATALOG_MANIFEST = "catalog.jsonl"
QUERIES_MANIFEST = "queries.jsonl"
IMAGES_SUBDIR = "images"

# Grid size for the synthetic images (kept tiny so the fixture is a few KB).
_IMG_SIZE = 48


@dataclass
class CatalogEntry:
    product_id: str
    image_id: str
    image_path: str
    is_reference: bool = True
    category: Optional[str] = None
    source_updated_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "is_reference": self.is_reference,
            "category": self.category,
            "source_updated_at": self.source_updated_at,
        }


@dataclass
class QueryEntry:
    query_image_path: str
    expected_product_id: Optional[str]
    category: Optional[str] = None
    in_catalog: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_image_path": self.query_image_path,
            "expected_product_id": self.expected_product_id,
            "category": self.category,
            "in_catalog": self.in_catalog,
        }


@dataclass
class EvalSet:
    catalog: list[CatalogEntry] = field(default_factory=list)
    queries: list[QueryEntry] = field(default_factory=list)


# --------------------------------------------------------------------------
# Manifest read/write
# --------------------------------------------------------------------------
def _write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_manifests(out_dir: str, eval_set: EvalSet) -> None:
    """Write ``catalog.jsonl`` + ``queries.jsonl`` into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    _write_jsonl(
        os.path.join(out_dir, CATALOG_MANIFEST),
        (e.to_dict() for e in eval_set.catalog),
    )
    _write_jsonl(
        os.path.join(out_dir, QUERIES_MANIFEST),
        (q.to_dict() for q in eval_set.queries),
    )


def load_manifests(dataset_dir: str) -> EvalSet:
    """Load ``catalog.jsonl`` + ``queries.jsonl`` from a dataset dir."""
    catalog = [
        CatalogEntry(
            product_id=r["product_id"],
            image_id=r["image_id"],
            image_path=r["image_path"],
            is_reference=bool(r.get("is_reference", True)),
            category=r.get("category"),
            source_updated_at=r.get("source_updated_at"),
        )
        for r in _read_jsonl(os.path.join(dataset_dir, CATALOG_MANIFEST))
    ]
    queries = [
        QueryEntry(
            query_image_path=r["query_image_path"],
            expected_product_id=r.get("expected_product_id"),
            category=r.get("category"),
            in_catalog=bool(r.get("in_catalog", True)),
        )
        for r in _read_jsonl(os.path.join(dataset_dir, QUERIES_MANIFEST))
    ]
    return EvalSet(catalog=catalog, queries=queries)


# --------------------------------------------------------------------------
# Assemble from an external export (production path)
# --------------------------------------------------------------------------
def assemble_from_rows(
    catalog_rows: Sequence[dict[str, Any]],
    query_rows: Sequence[dict[str, Any]],
) -> EvalSet:
    """Assemble an :class:`EvalSet` from external export rows.

    Catalog rows require ``product_id``, ``image_id``, ``image_path``; query
    rows require ``query_image_path`` and (for positives) ``expected_product_id``.
    Rows without a resolvable ``product_id`` mapping are skipped by the caller
    (never guessed) — mirroring the ``catalog_sync`` unmapped-image contract.
    """
    catalog = [
        CatalogEntry(
            product_id=str(r["product_id"]),
            image_id=str(r["image_id"]),
            image_path=str(r["image_path"]),
            is_reference=bool(r.get("is_reference", True)),
            category=r.get("category"),
            source_updated_at=r.get("source_updated_at"),
        )
        for r in catalog_rows
        if r.get("product_id") and r.get("image_id") and r.get("image_path")
    ]
    queries = [
        QueryEntry(
            query_image_path=str(r["query_image_path"]),
            expected_product_id=(
                str(r["expected_product_id"])
                if r.get("expected_product_id")
                else None
            ),
            category=r.get("category"),
            in_catalog=bool(r.get("in_catalog", bool(r.get("expected_product_id")))),
        )
        for r in query_rows
        if r.get("query_image_path")
    ]
    return EvalSet(catalog=catalog, queries=queries)


def assemble_from_csv(catalog_csv: str, queries_csv: str) -> EvalSet:
    """Assemble an :class:`EvalSet` from two CSV exports."""
    with open(catalog_csv, encoding="utf-8") as fh:
        catalog_rows = list(csv.DictReader(fh))
    with open(queries_csv, encoding="utf-8") as fh:
        query_rows = list(csv.DictReader(fh))
    return assemble_from_rows(catalog_rows, query_rows)


# --------------------------------------------------------------------------
# Synthetic fixture generation
# --------------------------------------------------------------------------
def _product_color(index: int, total: int) -> tuple[int, int, int]:
    """Evenly-spaced hue -> distinct RGB so products separate cleanly."""
    hue = (index / max(total, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
    return int(r * 255), int(g * 255), int(b * 255)


def _jittered(color: tuple[int, int, int], seed: int, amount: int) -> tuple[int, int, int]:
    """Deterministic small per-image jitter to simulate photo variation."""
    import random

    rng = random.Random(seed)
    return tuple(
        max(0, min(255, c + rng.randint(-amount, amount))) for c in color
    )  # type: ignore[return-value]


def _make_image(color: tuple[int, int, int], marker_index: int, size: int = _IMG_SIZE):
    """Solid product color + a small product-specific marker block.

    The marker (a lighter square whose position depends on the product) gives
    each product a distinct spatial signature so downscaled fake-encoder
    vectors separate even if two hues are close.
    """
    from PIL import Image

    img = Image.new("RGB", (size, size), color)
    px = img.load()
    # Marker: an 8x8 lighter block at a product-dependent grid cell.
    cells = size // 8
    cx = (marker_index % cells) * 8
    cy = (marker_index // cells % cells) * 8
    light = tuple(min(255, c + 70) for c in color)
    for y in range(cy, min(cy + 8, size)):
        for x in range(cx, min(cx + 8, size)):
            px[x, y] = light
    return img


def generate_synthetic_dataset(
    out_dir: str,
    n_products: int = 6,
    catalog_per_product: int = 2,
    queries_per_product: int = 2,
    out_of_catalog: int = 3,
    write_readme: bool = True,
) -> EvalSet:
    """Generate a deterministic synthetic labeled eval set under ``out_dir``.

    Creates tiny PNGs (products = distinct hue + spatial marker; query photos =
    same product with small jitter), both manifests, and a README. Fully
    deterministic so the committed fixture is stable and CI-reproducible.
    """
    images_dir = os.path.join(out_dir, IMAGES_SUBDIR)
    os.makedirs(images_dir, exist_ok=True)

    eval_set = EvalSet()
    categories = ["analgesic", "antibiotic", "antacid", "antihistamine", "vitamin", "topical"]

    for p in range(n_products):
        pid = f"P{p:04d}"
        category = categories[p % len(categories)]
        base = _product_color(p, n_products)
        for c in range(catalog_per_product):
            img = _make_image(_jittered(base, seed=1000 + p * 10 + c, amount=6), marker_index=p)
            rel = os.path.join(IMAGES_SUBDIR, f"{pid}-ref-{c}.png")
            img.save(os.path.join(out_dir, rel))
            eval_set.catalog.append(
                CatalogEntry(
                    product_id=pid,
                    image_id=f"{pid}-ref-{c}",
                    image_path=rel,
                    is_reference=True,
                    category=category,
                    source_updated_at="2026-07-01T00:00:00Z",
                )
            )
        for q in range(queries_per_product):
            # Query photos: same product color + marker, larger jitter (photo noise).
            img = _make_image(_jittered(base, seed=5000 + p * 10 + q, amount=18), marker_index=p)
            rel = os.path.join(IMAGES_SUBDIR, f"{pid}-q{q}.png")
            img.save(os.path.join(out_dir, rel))
            eval_set.queries.append(
                QueryEntry(
                    query_image_path=rel,
                    expected_product_id=pid,
                    category=category,
                    in_catalog=True,
                )
            )

    # Out-of-catalog negatives: colors/markers not present in the catalog.
    for k in range(out_of_catalog):
        # Use a grey-ish random image well away from the saturated product hues.
        import random

        rng = random.Random(90000 + k)
        color = (rng.randint(60, 200),) * 3
        img = _make_image(color, marker_index=n_products + k, size=_IMG_SIZE)
        rel = os.path.join(IMAGES_SUBDIR, f"OOC-{k}.png")
        img.save(os.path.join(out_dir, rel))
        eval_set.queries.append(
            QueryEntry(
                query_image_path=rel,
                expected_product_id=None,
                category="out_of_catalog",
                in_catalog=False,
            )
        )

    write_manifests(out_dir, eval_set)
    if write_readme:
        _write_readme(out_dir, eval_set)
    return eval_set


def _write_readme(out_dir: str, eval_set: EvalSet) -> None:
    n_pos = sum(1 for q in eval_set.queries if q.in_catalog)
    n_neg = sum(1 for q in eval_set.queries if not q.in_catalog)
    readme = f"""# Eval dataset (sample fixture)

Small **synthetic** labeled photo->product eval set generated by
`eval/build_eval_set.py`. It lets `eval/run_eval.py` run end-to-end in CI with
the deterministic `--fake-encoder` and an in-memory Qdrant — **no real catalog
data, no model downloads, no GPU**.

## Contents
- `{IMAGES_SUBDIR}/` — tiny synthetic PNGs ({_IMG_SIZE}x{_IMG_SIZE}px). Each
  product = a distinct hue + a product-specific marker block. Query photos are
  the same product with small colour jitter (simulated photo noise).
- `{CATALOG_MANIFEST}` — {len(eval_set.catalog)} reference catalog images (indexed into Qdrant).
- `{QUERIES_MANIFEST}` — {len(eval_set.queries)} labeled query photos
  ({n_pos} in-catalog positives, {n_neg} out-of-catalog negatives).

## Manifest format
`{CATALOG_MANIFEST}` (one JSON object per line):
```json
{{"product_id": "P0000", "image_id": "P0000-ref-0", "image_path": "images/P0000-ref-0.png", "is_reference": true, "category": "analgesic", "source_updated_at": "2026-07-01T00:00:00Z"}}
```
`{QUERIES_MANIFEST}` (one JSON object per line):
```json
{{"query_image_path": "images/P0000-q0.png", "expected_product_id": "P0000", "category": "analgesic", "in_catalog": true}}
```
Fields:
- `query_image_path` / `image_path` — path **relative to this dataset dir**.
- `expected_product_id` — correct product; `null` for out-of-catalog negatives
  (`in_catalog: false`), which drive false-positive / weak-match calibration.
- `category` — optional grouping label.
- `is_reference` — reference-image flag (Task 2 payload).
- `source_updated_at` — optional catalog timestamp for index-freshness checks.

## Regenerate
```bash
python eval/build_eval_set.py --out eval/dataset --synthetic
```
The generator is deterministic, so regeneration reproduces the committed files.

## Run the harness
```bash
python eval/run_eval.py --dataset eval/dataset --fake-encoder
```
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build/assemble a labeled eval set.")
    parser.add_argument("--out", default="eval/dataset", help="Output dataset directory.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="Generate the synthetic sample fixture.")
    mode.add_argument("--from-csv", nargs=2, metavar=("CATALOG_CSV", "QUERIES_CSV"),
                      help="Assemble manifests from two external CSV exports.")
    parser.add_argument("--n-products", type=int, default=6)
    parser.add_argument("--catalog-per-product", type=int, default=2)
    parser.add_argument("--queries-per-product", type=int, default=2)
    parser.add_argument("--out-of-catalog", type=int, default=3)
    args = parser.parse_args(argv)

    if args.synthetic:
        es = generate_synthetic_dataset(
            args.out,
            n_products=args.n_products,
            catalog_per_product=args.catalog_per_product,
            queries_per_product=args.queries_per_product,
            out_of_catalog=args.out_of_catalog,
        )
        print(
            f"Wrote synthetic eval set to {args.out}: "
            f"{len(es.catalog)} catalog images, {len(es.queries)} queries."
        )
    else:
        catalog_csv, queries_csv = args.from_csv
        es = assemble_from_csv(catalog_csv, queries_csv)
        write_manifests(args.out, es)
        print(
            f"Assembled eval set into {args.out}: "
            f"{len(es.catalog)} catalog images, {len(es.queries)} queries."
        )


if __name__ == "__main__":
    main()
