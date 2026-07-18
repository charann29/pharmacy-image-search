-- 0001_init.sql — initial schema for the image-search catalog metadata store.
--
-- Target: Cloudflare D1 (SQLite-compatible). Apply locally with:
--   wrangler d1 execute <DB> --local --file db/migrations/0001_init.sql
--
-- Design notes:
--   * products holds catalog metadata ONLY. Price / commerce fields are NOT
--     stored here — price is fetched from the existing commerce API at
--     response time (see plan Task 8). A denormalized price cache, if ever
--     wanted, must arrive as a separate migration.
--   * product_images.deleted_at supports soft-delete: catalog_sync marks rows
--     deleted, and reconcile.py clears the corresponding Qdrant vectors before
--     any hard delete happens.
--   * embedding_map.image_id uses ON DELETE RESTRICT so an image row cannot be
--     hard-deleted while its vector mapping still exists (prevents orphaned
--     vectors). Images are soft-deleted, then reconcile.py removes vectors,
--     then rows may be hard-deleted.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- products: catalog metadata (no price/commerce fields).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
    product_id   TEXT PRIMARY KEY,
    sku          TEXT,
    name         TEXT NOT NULL,
    manufacturer TEXT,
    strength     TEXT,
    barcode      TEXT,
    active       INTEGER NOT NULL DEFAULT 1,   -- boolean 0/1
    updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Barcode lookups (tie-breaking for near-identical generics, Task 9).
CREATE INDEX IF NOT EXISTS idx_products_barcode ON products (barcode);

-- ---------------------------------------------------------------------------
-- product_images: ImageKit inventory, one row per catalog image.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS product_images (
    image_id          TEXT PRIMARY KEY,
    product_id        TEXT NOT NULL,
    imagekit_file_id  TEXT,
    imagekit_url      TEXT,
    is_reference      INTEGER NOT NULL DEFAULT 0,  -- boolean 0/1
    source_updated_at TEXT,
    deleted_at        TEXT,                        -- NULL = live; set = soft-deleted
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);

CREATE INDEX IF NOT EXISTS idx_product_images_product_id ON product_images (product_id);

-- ---------------------------------------------------------------------------
-- embedding_map: maps Qdrant vector ids <-> image/product, per encoder.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embedding_map (
    vector_id     TEXT PRIMARY KEY,        -- deterministic uuid5 (see qdrant_client.py)
    image_id      TEXT NOT NULL,
    product_id    TEXT NOT NULL,
    encoder       TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    indexed_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- RESTRICT: cannot hard-delete an image row while its vector mapping lives.
    FOREIGN KEY (image_id) REFERENCES product_images (image_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_embedding_map_product_id ON embedding_map (product_id);
CREATE INDEX IF NOT EXISTS idx_embedding_map_encoder ON embedding_map (encoder);
