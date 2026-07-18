import { describe, it, expect, beforeAll, beforeEach } from "vitest";
import { env as testEnv, fetchMock, createExecutionContext, waitOnExecutionContext } from "cloudflare:test";
import worker from "../src/index";
import type { Env, RateLimitBinding } from "../src/types";
import { computeSignature } from "../src/ml_client";

// NOTE: `fetchMock` interceptors persist for the whole test *file* (they are
// reset per file, not per test), and undici matches the first *unconsumed*
// interceptor for a given origin+path in registration order. To keep tests
// deterministic we (a) never register an interceptor a test won't consume, and
// (b) place the timeout tests (whose interceptors are intentionally aborted and
// therefore left unconsumed) LAST so their leftovers cannot pollute others.

const DB = (testEnv as any).DB as D1Database;
const ML = "https://ml.example.com";
const TEXT = "https://text.example.com";
const COMMERCE = "https://commerce.example.com";

const okLimiter: RateLimitBinding = { async limit() { return { success: true }; } };

function baseEnv(overrides: Partial<Env> = {}): Env {
  return {
    DB,
    RATE_LIMITER: okLimiter,
    ML_SERVICE_URL: ML,
    ML_SERVICE_SHARED_SECRET: "key",
    TEXT_SEARCH_URL: `${TEXT}/search`,
    TEXT_SEARCH_AUTH: "Bearer t",
    COMMERCE_API_URL: `${COMMERCE}/prices`,
    COMMERCE_API_AUTH: "Bearer c",
    IMAGE_SEARCH_ENABLED: "true",
    HYBRID_FUSION_ENABLED: "true",
    ...overrides,
  } as Env;
}

function post(body: BodyInit = new Uint8Array([1, 2, 3, 4])): Request {
  return new Request("https://w/search", {
    method: "POST",
    body,
    headers: { "cf-connecting-ip": "5.5.5.5" },
  });
}

async function run(env: Env, request: Request): Promise<Response> {
  const ctx = createExecutionContext();
  const res = await worker.fetch(request, env, ctx);
  await waitOnExecutionContext(ctx);
  return res;
}

beforeAll(async () => {
  await DB.prepare(
    `CREATE TABLE IF NOT EXISTS products (product_id TEXT PRIMARY KEY, sku TEXT, name TEXT, manufacturer TEXT, strength TEXT, barcode TEXT, active INTEGER NOT NULL DEFAULT 1, updated_at TEXT)`,
  ).run();
  await DB.prepare(
    `CREATE TABLE IF NOT EXISTS product_images (image_id TEXT PRIMARY KEY, product_id TEXT NOT NULL, imagekit_file_id TEXT, imagekit_url TEXT, is_reference INTEGER NOT NULL DEFAULT 0, source_updated_at TEXT, deleted_at TEXT, created_at TEXT)`,
  ).run();
  await DB.prepare("DELETE FROM products").run();
  await DB.prepare("DELETE FROM product_images").run();
  for (const id of ["A", "B", "Z"]) {
    await DB.prepare(
      "INSERT INTO products (product_id, name, manufacturer, strength, sku, barcode, active) VALUES (?,?,?,?,?,?,1)",
    ).bind(id, `name-${id}`, "mfr", "650mg", `sku-${id}`, `bc-${id}`).run();
    await DB.prepare(
      "INSERT INTO product_images (image_id, product_id, imagekit_url, is_reference, created_at) VALUES (?,?,?,1,?)",
    ).bind(`img-${id}`, id, `https://ik/${id}.jpg`, "2024-01-01T00:00:00Z").run();
  }
  // Near-identical generics for the tie-break test: same name/strength, distinct
  // numeric barcodes so OCR-derived barcode can disambiguate.
  await DB.prepare(
    "INSERT INTO products (product_id, name, manufacturer, strength, sku, barcode, active) VALUES ('G1','Generic Paracetamol','AcmeCo','650mg','sku-G1','8901111111',1)",
  ).run();
  await DB.prepare(
    "INSERT INTO products (product_id, name, manufacturer, strength, sku, barcode, active) VALUES ('G2','Generic Paracetamol','BetaCorp','650mg','sku-G2','8902222222',1)",
  ).run();
  for (const id of ["G1", "G2"]) {
    await DB.prepare(
      "INSERT INTO product_images (image_id, product_id, imagekit_url, is_reference, created_at) VALUES (?,?,?,1,?)",
    ).bind(`img-${id}`, id, `https://ik/${id}.jpg`, "2024-01-01T00:00:00Z").run();
  }
});

beforeEach(() => {
  fetchMock.activate();
  fetchMock.disableNetConnect();
});

function mockMlSearch(products: Array<{ product_id: string; score: number }>, weak = false, capture?: (body: string) => void) {
  fetchMock
    .get(ML)
    .intercept({ path: "/search", method: "POST" })
    .reply(200, (opts: any) => {
      capture?.(bufToString(opts.body));
      return JSON.stringify({ products, weak_visual_match: weak });
    });
}

function mockMlOcr(candidates: string[], capture?: (body: string) => void) {
  fetchMock
    .get(ML)
    .intercept({ path: "/ocr", method: "POST" })
    .reply(200, (opts: any) => {
      capture?.(bufToString(opts.body));
      return JSON.stringify({ candidates, tokens: [] });
    });
}

function mockText(results: Array<{ product_id: string; score: number }>) {
  fetchMock
    .get(TEXT)
    .intercept({ path: "/search", method: "POST" })
    .reply(200, JSON.stringify({ results }));
}

function mockCommerce(prices: Record<string, unknown>) {
  fetchMock
    .get(COMMERCE)
    .intercept({ path: "/prices", method: "POST" })
    .reply(200, JSON.stringify({ prices }));
}

describe("orchestration — happy path", () => {
  it("reads photo once, fans out to /search + /ocr, fuses, hydrates, prices", async () => {
    let searchBody = "";
    let ocrBody = "";
    mockMlSearch([{ product_id: "A", score: 0.9 }, { product_id: "B", score: 0.7 }], false, (b) => (searchBody = b));
    mockMlOcr(["crocin", "650"], (b) => (ocrBody = b));
    mockText([{ product_id: "B", score: 0.8 }]);
    mockCommerce({ A: { price: 12, currency: "INR", in_stock: true }, B: { price: 5, currency: "INR", in_stock: false } });

    const res = await run(baseEnv(), post(new Uint8Array([9, 8, 7, 6])));
    expect(res.status).toBe(200);
    const body = (await res.json()) as any;

    // Same bytes fanned out to BOTH endpoints.
    expect(searchBody).toBe(ocrBody);
    expect(searchBody).toBe(String.fromCharCode(9, 8, 7, 6));

    expect(body.ocr_candidates).toEqual(["crocin", "650"]);
    expect(body.products.length).toBeGreaterThan(0);
    const a = body.products.find((p: any) => p.product_id === "A");
    expect(a).toMatchObject({ name: "name-A", image_url: "https://ik/A.jpg", price: 12, currency: "INR", in_stock: true });
    expect(body.fusion.mode).toBe("rrf");
  });

  it("weak_visual_match prefers text", async () => {
    mockMlSearch([{ product_id: "A", score: 0.2 }], true);
    mockMlOcr(["dolo"]);
    // Text ranks Z first; the second text item (W) is not in the catalog so it
    // is dropped in hydration and cannot dilute the ranking.
    mockText([{ product_id: "Z", score: 0.95 }, { product_id: "W", score: 0.3 }]);
    mockCommerce({});

    const res = await run(baseEnv({ WEAK_VISUAL_IMAGE_MULTIPLIER: "0.05" }), post());
    const body = (await res.json()) as any;
    expect(body.weak_visual_match).toBe(true);
    expect(body.products[0].product_id).toBe("Z");
  });

  it("tie-breaks near-identical generics using OCR barcode/manufacturer", async () => {
    // Visual scores put G1 slightly ahead of G2 but within the tie-break delta.
    // OCR captures G2's barcode + manufacturer -> G2 should be promoted to #1.
    mockMlSearch([
      { product_id: "G1", score: 0.90 },
      { product_id: "G2", score: 0.88 },
    ]);
    mockMlOcr(["Generic Paracetamol 650mg", "BetaCorp", "8902222222"]);
    mockText([]); // isolate the visual + tie-break path
    mockCommerce({});

    const res = await run(
      baseEnv({ HYBRID_FUSION_ENABLED: "false", TIE_BREAK_DELTA: "0.2" }),
      post(),
    );
    const body = (await res.json()) as any;
    expect(body.products.map((p: any) => p.product_id)).toEqual(["G2", "G1"]);
  });
});

describe("orchestration — feature flags", () => {
  it("flags off yields PURE text-only behavior with ZERO ML dependency", async () => {
    // Register NO ML interceptors at all. With net-connect disabled, ANY ML
    // call (/search or /ocr) would throw and fail the test — proving zero ML
    // fan-out. We also unset ML_SERVICE_SHARED_SECRET to prove the path works
    // even when the ML service is unconfigured/down (additive rollback).
    mockText([{ product_id: "Z", score: 0.9 }, { product_id: "B", score: 0.5 }]);
    mockCommerce({});

    const res = await run(
      baseEnv({
        IMAGE_SEARCH_ENABLED: "false",
        HYBRID_FUSION_ENABLED: "false",
        ML_SERVICE_SHARED_SECRET: undefined,
        ML_SERVICE_URL: "http://ml-is-down.invalid",
      }),
      // Text query supplied via request (query param), not via OCR.
      new Request("https://w/search?q=dolo%20650", {
        method: "POST",
        body: new Uint8Array([1, 2, 3, 4]),
        headers: { "cf-connecting-ip": "5.5.5.5" },
      }),
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as any;
    expect(body.fusion.mode).toBe("text-only");
    expect(body.ocr_candidates).toEqual([]);
    expect(body.products.map((p: any) => p.product_id)).toEqual(["Z", "B"]);
  });

  it("flags off with no text query returns empty results (no ML call)", async () => {
    // No text query and no ML calls -> empty result set, still HTTP 200.
    const res = await run(
      baseEnv({
        IMAGE_SEARCH_ENABLED: "false",
        ML_SERVICE_SHARED_SECRET: undefined,
      }),
      post(),
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as any;
    expect(body.fusion.mode).toBe("text-only");
    expect(body.products).toEqual([]);
  });

  it("hybrid off yields image-only ranking", async () => {
    mockMlSearch([{ product_id: "B", score: 0.9 }, { product_id: "A", score: 0.4 }]);
    mockMlOcr(["q"]);
    mockText([{ product_id: "A", score: 1 }]);
    mockCommerce({});

    const res = await run(baseEnv({ HYBRID_FUSION_ENABLED: "false" }), post());
    const body = (await res.json()) as any;
    expect(body.fusion.mode).toBe("image-only");
    expect(body.products.map((p: any) => p.product_id)).toEqual(["B", "A"]);
  });
});

describe("orchestration — guards", () => {
  it("rate limiter blocks over-limit with 429", async () => {
    const blockLimiter: RateLimitBinding = { async limit() { return { success: false }; } };
    const res = await run(baseEnv({ RATE_LIMITER: blockLimiter }), post());
    expect(res.status).toBe(429);
  });

  it("oversize upload -> 413", async () => {
    const res = await run(baseEnv({ MAX_UPLOAD_BYTES: "2" }), post(new Uint8Array([1, 2, 3, 4, 5])));
    expect(res.status).toBe(413);
  });

  it("empty upload -> 400", async () => {
    const res = await run(baseEnv(), post(new Uint8Array([])));
    expect(res.status).toBe(400);
  });

  it("missing ML signing key fails closed -> 502", async () => {
    // No fetch mocks needed: ml_client throws before issuing a request.
    const res = await run(baseEnv({ ML_SERVICE_SHARED_SECRET: undefined }), post());
    expect(res.status).toBe(502);
    const body = (await res.json()) as any;
    expect(body.error).toBe("ml_auth_failed");
  });

  it("signs ML requests with a valid path-bound HMAC", async () => {
    let sig = "";
    let ts = "";
    mockMlSearch([], false, () => {});
    // Overwrite search capture with header capture via a dedicated interceptor.
    fetchMock
      .get(ML)
      .intercept({ path: "/ocr", method: "POST" })
      .reply(200, (opts: any) => {
        const h = normHeaders(opts.headers);
        sig = h["x-signature"];
        ts = h["x-timestamp"];
        return JSON.stringify({ candidates: [], tokens: [] });
      });
    mockText([]);
    mockCommerce({});
    await run(baseEnv(), post());
    expect(ts).toBeTruthy();
    expect(await computeSignature("key", ts, "POST", "/ocr")).toBe(sig);
  });
});

// Timeout tests LAST: their aborted interceptors are left unconsumed and would
// otherwise leak into later tests (see file-level note).
describe("orchestration — graceful degradation on timeout", () => {
  it("commerce timeout omits price without failing the response", async () => {
    mockMlSearch([{ product_id: "A", score: 0.9 }]);
    mockMlOcr(["x"]);
    mockText([]);
    fetchMock
      .get(COMMERCE)
      .intercept({ path: "/prices", method: "POST" })
      .reply(200, JSON.stringify({ prices: {} }))
      .delay(300);

    const res = await run(baseEnv({ COMMERCE_TIMEOUT_MS: "20" }), post());
    expect(res.status).toBe(200);
    const body = (await res.json()) as any;
    const a = body.products.find((p: any) => p.product_id === "A");
    expect(a.price).toBeNull();
    expect(a.in_stock).toBeNull();
  });

  it("text-adapter timeout degrades to image-only ranking", async () => {
    mockMlSearch([{ product_id: "A", score: 0.9 }, { product_id: "B", score: 0.6 }]);
    mockMlOcr(["q"]);
    fetchMock
      .get(TEXT)
      .intercept({ path: "/search", method: "POST" })
      .reply(200, JSON.stringify({ results: [{ product_id: "Z", score: 1 }] }))
      .delay(300);
    mockCommerce({});

    const res = await run(baseEnv({ TEXT_SEARCH_TIMEOUT_MS: "20" }), post());
    const body = (await res.json()) as any;
    const ids = body.products.map((p: any) => p.product_id);
    // Text (Z) timed out -> only image products A,B present, A first.
    expect(ids).toEqual(["A", "B"]);
  });
});

function bufToString(body: unknown): string {
  if (typeof body === "string") return body;
  if (body instanceof ArrayBuffer) return String.fromCharCode(...new Uint8Array(body));
  if (body instanceof Uint8Array) return String.fromCharCode(...body);
  return String(body);
}

function normHeaders(headers: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (Array.isArray(headers)) {
    for (let i = 0; i < headers.length; i += 2) out[String(headers[i]).toLowerCase()] = String(headers[i + 1]);
  } else if (headers && typeof headers === "object") {
    for (const [k, v] of Object.entries(headers as Record<string, unknown>)) out[k.toLowerCase()] = String(v);
  }
  return out;
}
