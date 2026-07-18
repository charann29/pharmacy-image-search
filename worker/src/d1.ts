/**
 * d1.ts — catalog metadata hydration via the D1 **binding** (`env.DB`).
 *
 * These queries use the bound `D1Database` (prepared statements), NOT the D1
 * REST API / Cloudflare API token. The REST path belongs only to the indexing
 * environment (catalog_sync / backfill); the Worker reads via its binding.
 */

import type { ProductMetadata } from "./types";

/**
 * Hydrate product metadata for a list of product_ids, joining a representative
 * `product_images.imagekit_url` per product.
 *
 * Representative image selection: prefer a reference image (`is_reference=1`),
 * then the most recently created live (non-soft-deleted) image. We compute this
 * with a correlated subquery so the join yields exactly one row per product.
 *
 * Returns a map keyed by product_id. Missing product_ids are simply absent.
 * Preserves nothing about ordering — callers own ranking.
 */
export async function hydrateProducts(
  db: D1Database,
  productIds: string[],
): Promise<Map<string, ProductMetadata>> {
  const result = new Map<string, ProductMetadata>();
  const ids = dedupe(productIds);
  if (ids.length === 0) return result;

  const placeholders = ids.map(() => "?").join(", ");
  // Representative image: reference-first, then newest live image.
  const sql = `
    SELECT
      p.product_id   AS product_id,
      p.sku          AS sku,
      p.name         AS name,
      p.manufacturer AS manufacturer,
      p.strength     AS strength,
      p.barcode      AS barcode,
      (
        SELECT pi.imagekit_url
        FROM product_images pi
        WHERE pi.product_id = p.product_id
          AND pi.deleted_at IS NULL
        ORDER BY pi.is_reference DESC, pi.created_at DESC
        LIMIT 1
      )              AS imagekit_url
    FROM products p
    WHERE p.product_id IN (${placeholders})
      AND p.active = 1
  `;

  const stmt = db.prepare(sql).bind(...ids);
  const { results } = await stmt.all<{
    product_id: string;
    sku: string | null;
    name: string | null;
    manufacturer: string | null;
    strength: string | null;
    barcode: string | null;
    imagekit_url: string | null;
  }>();

  for (const row of results ?? []) {
    result.set(row.product_id, {
      product_id: row.product_id,
      sku: row.sku ?? null,
      name: row.name ?? null,
      manufacturer: row.manufacturer ?? null,
      strength: row.strength ?? null,
      barcode: row.barcode ?? null,
      imagekit_url: row.imagekit_url ?? null,
    });
  }
  return result;
}

/** Lookup a single product by barcode (tie-breaker helper, Task 9). */
export async function findProductByBarcode(
  db: D1Database,
  barcode: string,
): Promise<ProductMetadata | null> {
  const sql = `
    SELECT product_id, sku, name, manufacturer, strength, barcode
    FROM products
    WHERE barcode = ? AND active = 1
    LIMIT 1
  `;
  const row = await db
    .prepare(sql)
    .bind(barcode)
    .first<{
      product_id: string;
      sku: string | null;
      name: string | null;
      manufacturer: string | null;
      strength: string | null;
      barcode: string | null;
    }>();
  if (!row) return null;
  return {
    product_id: row.product_id,
    sku: row.sku ?? null,
    name: row.name ?? null,
    manufacturer: row.manufacturer ?? null,
    strength: row.strength ?? null,
    barcode: row.barcode ?? null,
    imagekit_url: null,
  };
}

function dedupe(items: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const item of items) {
    if (item && !seen.has(item)) {
      seen.add(item);
      out.push(item);
    }
  }
  return out;
}
