import { describe, it, expect, beforeAll } from "vitest";
import { env } from "cloudflare:test";
import { hydrateProducts, findProductByBarcode } from "../src/d1";

const DB = (env as any).DB as D1Database;

const SCHEMA = [
  `CREATE TABLE IF NOT EXISTS products (
     product_id TEXT PRIMARY KEY, sku TEXT, name TEXT, manufacturer TEXT,
     strength TEXT, barcode TEXT, active INTEGER NOT NULL DEFAULT 1,
     updated_at TEXT
   )`,
  `CREATE TABLE IF NOT EXISTS product_images (
     image_id TEXT PRIMARY KEY, product_id TEXT NOT NULL, imagekit_file_id TEXT,
     imagekit_url TEXT, is_reference INTEGER NOT NULL DEFAULT 0,
     source_updated_at TEXT, deleted_at TEXT, created_at TEXT
   )`,
];

beforeAll(async () => {
  for (const stmt of SCHEMA) await DB.prepare(stmt).run();
  await DB.prepare("DELETE FROM products").run();
  await DB.prepare("DELETE FROM product_images").run();

  await DB.batch([
    DB.prepare(
      "INSERT INTO products (product_id, sku, name, manufacturer, strength, barcode, active) VALUES (?,?,?,?,?,?,1)",
    ).bind("A", "SKU-A", "Crocin", "GSK", "650mg", "111", ),
    DB.prepare(
      "INSERT INTO products (product_id, sku, name, manufacturer, strength, barcode, active) VALUES (?,?,?,?,?,?,1)",
    ).bind("B", "SKU-B", "Dolo", "Micro", "650mg", "222"),
    DB.prepare(
      "INSERT INTO products (product_id, sku, name, manufacturer, strength, barcode, active) VALUES (?,?,?,?,?,?,0)",
    ).bind("C", "SKU-C", "Inactive", "X", "1mg", "333"),
  ]);

  await DB.batch([
    // A: two images, one reference (should win) + one newer non-reference.
    DB.prepare(
      "INSERT INTO product_images (image_id, product_id, imagekit_url, is_reference, created_at, deleted_at) VALUES (?,?,?,?,?,NULL)",
    ).bind("a1", "A", "https://ik/a-ref.jpg", 1, "2020-01-01T00:00:00Z"),
    DB.prepare(
      "INSERT INTO product_images (image_id, product_id, imagekit_url, is_reference, created_at, deleted_at) VALUES (?,?,?,?,?,NULL)",
    ).bind("a2", "A", "https://ik/a-new.jpg", 0, "2024-01-01T00:00:00Z"),
    // B: only a soft-deleted image -> should hydrate with null url.
    DB.prepare(
      "INSERT INTO product_images (image_id, product_id, imagekit_url, is_reference, created_at, deleted_at) VALUES (?,?,?,?,?,?)",
    ).bind("b1", "B", "https://ik/b.jpg", 0, "2024-01-01T00:00:00Z", "2024-06-01T00:00:00Z"),
  ]);
});

describe("d1 binding join", () => {
  it("hydrates metadata with a representative reference image", async () => {
    const map = await hydrateProducts(DB, ["A", "B"]);
    expect(map.get("A")).toMatchObject({
      product_id: "A",
      name: "Crocin",
      manufacturer: "GSK",
      strength: "650mg",
      sku: "SKU-A",
      barcode: "111",
      imagekit_url: "https://ik/a-ref.jpg", // reference preferred over newer
    });
    // B has only a soft-deleted image -> url null but metadata present.
    expect(map.get("B")?.imagekit_url).toBeNull();
    expect(map.get("B")?.name).toBe("Dolo");
  });

  it("excludes inactive products", async () => {
    const map = await hydrateProducts(DB, ["C"]);
    expect(map.has("C")).toBe(false);
  });

  it("dedupes input ids and returns empty on empty input", async () => {
    const map = await hydrateProducts(DB, []);
    expect(map.size).toBe(0);
    const map2 = await hydrateProducts(DB, ["A", "A", "A"]);
    expect(map2.size).toBe(1);
  });

  it("finds a product by barcode", async () => {
    const p = await findProductByBarcode(DB, "222");
    expect(p?.product_id).toBe("B");
    const none = await findProductByBarcode(DB, "999");
    expect(none).toBeNull();
  });
});
