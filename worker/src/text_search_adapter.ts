/**
 * text_search_adapter.ts — adapter to the EXISTING keyword + semantic text
 * search (treated as a black box).
 *
 * Contract: `search(queryText) -> [{product_id, score, rank}]`.
 * Configured via `TEXT_SEARCH_URL` + `TEXT_SEARCH_AUTH`.
 *
 * Graceful degradation: on error, timeout, missing config, or malformed
 * response we return `[]` and NEVER throw — the image-search request must not
 * fail because the text tier is unavailable.
 */

import type { Env, RankedItem } from "./types";

const DEFAULT_TIMEOUT_MS = 2500;

function timeoutMs(env: Env): number {
  const parsed = Number(env.TEXT_SEARCH_TIMEOUT_MS);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_TIMEOUT_MS;
}

/**
 * Query the existing text search with an OCR-derived string.
 *
 * Request  (POST JSON): `{ "query": <string>, "limit": <number> }`
 * Response (JSON): either
 *   - `{ "results": [{ "product_id": string, "score": number }] }`, or
 *   - a bare array `[{ "product_id": string, "score": number }]`.
 * Rank is assigned by result order (1-based) if not provided.
 */
export async function search(
  env: Env,
  queryText: string,
  limit = 50,
): Promise<RankedItem[]> {
  const url = (env.TEXT_SEARCH_URL || "").trim();
  const query = (queryText || "").trim();
  if (!url || !query) return [];

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs(env));
  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (env.TEXT_SEARCH_AUTH) {
      headers["Authorization"] = env.TEXT_SEARCH_AUTH;
    }
    const resp = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({ query, limit }),
      signal: controller.signal,
    });
    if (!resp.ok) return [];
    const json = (await resp.json()) as unknown;
    return normalize(json);
  } catch {
    // Timeout / network / parse error -> degrade to empty (image-only).
    return [];
  } finally {
    clearTimeout(timer);
  }
}

function normalize(json: unknown): RankedItem[] {
  let rows: unknown[];
  if (Array.isArray(json)) {
    rows = json;
  } else if (json && typeof json === "object" && Array.isArray((json as any).results)) {
    rows = (json as any).results;
  } else {
    return [];
  }

  const out: RankedItem[] = [];
  let rank = 1;
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const r = row as Record<string, unknown>;
    const productId =
      typeof r.product_id === "string"
        ? r.product_id
        : typeof r.productId === "string"
          ? (r.productId as string)
          : r.product_id != null
            ? String(r.product_id)
            : null;
    if (!productId) continue;
    const score = typeof r.score === "number" ? r.score : 0;
    const providedRank = typeof r.rank === "number" ? r.rank : rank;
    out.push({ product_id: productId, score, rank: providedRank });
    rank++;
  }
  return out;
}
