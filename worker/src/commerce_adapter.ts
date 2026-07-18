/**
 * commerce_adapter.ts — adapter to the EXISTING commerce API for pricing /
 * availability (treated as a black box).
 *
 * Contract: `getPrices(productIds[]) -> { product_id: {price, currency, in_stock} }`.
 * Configured via `COMMERCE_API_URL` + `COMMERCE_API_AUTH`.
 *
 * BATCH hydration: all product ids are fetched in a single request.
 *
 * Graceful degradation: on error, timeout, missing config, or malformed
 * response we return `{}` (price omitted / availability unknown) and NEVER
 * throw — a commerce outage must not fail the search response.
 */

import type { Env, PriceMap, PriceInfo } from "./types";
import { envNumber, dedupe } from "./util";

const DEFAULT_TIMEOUT_MS = 2500;

function timeoutMs(env: Env): number {
  return envNumber(env.COMMERCE_TIMEOUT_MS, DEFAULT_TIMEOUT_MS, {
    positive: true,
  });
}

/**
 * Batch-fetch pricing/availability for the given product ids.
 *
 * Request  (POST JSON): `{ "product_ids": string[] }`
 * Response (JSON): either
 *   - `{ "prices": { "<product_id>": {price, currency, in_stock} } }`, or
 *   - `{ "<product_id>": {price, currency, in_stock} }`, or
 *   - `[{ "product_id": string, price, currency, in_stock }]`.
 */
export async function getPrices(
  env: Env,
  productIds: string[],
): Promise<PriceMap> {
  const url = (env.COMMERCE_API_URL || "").trim();
  const ids = dedupe(productIds);
  if (!url || ids.length === 0) return {};

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs(env));
  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (env.COMMERCE_API_AUTH) {
      headers["Authorization"] = env.COMMERCE_API_AUTH;
    }
    const resp = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({ product_ids: ids }),
      signal: controller.signal,
    });
    if (!resp.ok) return {};
    const json = (await resp.json()) as unknown;
    return normalize(json);
  } catch {
    // Timeout / network / parse error -> omit price, mark unavailable.
    return {};
  } finally {
    clearTimeout(timer);
  }
}

function coercePriceInfo(raw: unknown): PriceInfo | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const price =
    typeof r.price === "number"
      ? r.price
      : typeof r.price === "string" && r.price.trim() !== "" && !isNaN(Number(r.price))
        ? Number(r.price)
        : null;
  const currency =
    typeof r.currency === "string" ? r.currency : null;
  const inStock =
    typeof r.in_stock === "boolean"
      ? r.in_stock
      : typeof r.inStock === "boolean"
        ? (r.inStock as boolean)
        : false;
  return { price, currency, in_stock: inStock };
}

function normalize(json: unknown): PriceMap {
  const out: PriceMap = {};
  if (!json) return out;

  // Array form: [{ product_id, price, currency, in_stock }]
  if (Array.isArray(json)) {
    for (const row of json) {
      if (!row || typeof row !== "object") continue;
      const r = row as Record<string, unknown>;
      const id = r.product_id != null ? String(r.product_id) : null;
      if (!id) continue;
      const info = coercePriceInfo(r);
      if (info) out[id] = info;
    }
    return out;
  }

  if (typeof json === "object") {
    // Nested under "prices".
    const source =
      (json as any).prices && typeof (json as any).prices === "object"
        ? (json as any).prices
        : json;
    for (const [id, raw] of Object.entries(source as Record<string, unknown>)) {
      const info = coercePriceInfo(raw);
      if (info) out[id] = info;
    }
  }
  return out;
}

