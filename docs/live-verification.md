# Live verification against the production catalog (2026-07-18)

The two managed dependencies and the full retrieval path were verified **live**
against the real account. All credentials were supplied via environment
variables only — no literal secret values appear in this repo. The probe is
**read-only** against production D1 and uses a **scratch** Qdrant collection that
is deleted afterward; nothing in the production catalog or ImageKit account was
mutated.

## What passed

| Check | Result |
|-------|--------|
| Cloudflare D1 auth + reachability | HTTP 200 — database `pharmacy-products-v2` |
| Live catalog size | **80,073 products**, **33,103 product_images** |
| ImageKit auth (`Authorization: Basic base64(PRIVATE_KEY + ":")`) | HTTP 200 (list endpoint) |
| ImageKit image fetch (original) | HTTP 200, valid JPEG |
| ImageKit transform (`?tr=w-512,h-512`, used by backfill) | HTTP 200, valid JPEG |
| End-to-end: live D1 read → real ImageKit fetch → embed → Qdrant index → query | **rank-1 self-match, score 1.0000, ~2.3 ms** |

Reproduce with `ml-service/scripts/live_e2e_probe.py` (requires
`CF_ACCOUNT_ID`, `CF_D1_DATABASE_ID`, `CF_D1_API_TOKEN` in the environment and a
Qdrant reachable at `localhost:6333`). It selects a handful of real rows, fetches
their real images, embeds them with a deterministic **fake** encoder (this
sandbox has no GPU, so real DINOv3/SigLIP2 weights cannot be loaded), indexes to
a scratch collection, verifies the query image's own product ranks #1, and drops
the scratch collection.

> Note observed during the probe: several sampled catalog entries currently point
> at a **byte-identical placeholder image** (e.g. product IDs 74822/74824/74825
> all 2881-byte JPEGs). Any content-based encoder will treat these as identical,
> so visual search cannot disambiguate them until real pack photos are uploaded.
> This is a data-quality issue in the catalog, not a pipeline defect.

## Schema mismatch — action required before backfill can run against prod

This project was built **greenfield** to the approved plan's schema. The **live**
catalog uses a different schema:

| Concept | This repo (plan / `db/migrations/0001_init.sql`) | Live `pharmacy-products-v2` |
|---------|---------------------------------------------------|-----------------------------|
| products PK | `product_id TEXT` | `id INTEGER` |
| product image URL | `product_images.imagekit_url` + `imagekit_file_id` | `product_images.url` only |
| image PK | `image_id TEXT` | `id INTEGER AUTOINCREMENT` |
| vector mapping | `embedding_map` table (per-encoder) | **does not exist** |
| commerce fields | none in `products` (fetched from commerce API) | `mrp/price/selling_price/...` inline |

The indexing scripts (`catalog_sync.py`, `backfill_index.py`, `reconcile.py`)
and `d1_client.py` therefore will **not** run end-to-end against the live catalog
as-is. Two options (a **decision for the catalog owner**, not something to be
applied silently to production):

1. **Adapter layer (recommended, non-destructive):** add a thin mapping in the
   indexing scripts that reads the live columns (`products.id`,
   `product_images.url`) and writes the vector mapping to a **new**
   `embedding_map` table only (additive; never touches `products` /
   `product_images`). No change to the live catalog schema.
2. **Migrate the catalog** to the plan schema. Higher risk — the presence of
   `products_bak` / `product_images_bak` suggests a prior risky migration — and
   unnecessary for image search. Not recommended.

No DDL/DML was run against production for either option; this document only
records the finding and the recommended path.
