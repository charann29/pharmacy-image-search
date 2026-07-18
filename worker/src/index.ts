/**
 * index.ts — image-search Worker router / orchestration.
 *
 * Image-search endpoint flow (POST /search) when IMAGE_SEARCH_ENABLED=true:
 *   1. rate-limit (per IP/token) via the Workers binding
 *   2. read the uploaded photo ONCE into an ArrayBuffer under a strict max-size
 *      limit (a request body stream cannot be consumed twice)
 *   3. send the SAME bytes to ML `/search` and `/ocr` in parallel
 *   4. forward OCR text to the text-search adapter
 *   5. fuse image + text results (RRF default / weighted; weak-match aware)
 *   6. hydrate product metadata from D1 (binding)
 *   7. near-tie tie-break using OCR name/strength + manufacturer + barcode
 *   8. batch-hydrate price/availability from the commerce adapter
 *   9. return a shaped JSON response
 *
 * Feature flags:
 *   - IMAGE_SEARCH_ENABLED  = "false" -> PURE text-search behavior with ZERO ML
 *     dependency: makes NO ML calls (neither /search nor /ocr). The text query
 *     comes from the request (`?q=` param or `X-Search-Text` header) and is sent
 *     straight to the existing text-search adapter. This is the additive
 *     rollback path — it works even if the ML service is down/unconfigured.
 *   - HYBRID_FUSION_ENABLED = "false" -> no fusion; use image-only ranking.
 */

import type {
  Env,
  RankedItem,
  FusedItem,
  MlSearchResponse,
  MlOcrResponse,
  ShapedProduct,
  SearchApiResponse,
  ProductMetadata,
  PriceMap,
} from "./types";
import { readUploadOnce, maybePersistToImageKit, UploadTooLargeError, EmptyUploadError } from "./upload";
import { checkRateLimit } from "./ratelimit";
import * as mlClient from "./ml_client";
import { MlAuthError } from "./ml_client";
import * as textAdapter from "./text_search_adapter";
import * as commerce from "./commerce_adapter";
import {
  fuse,
  fusionParamsFromEnv,
  rankMlHits,
  applyTieBreak,
  tieBreakDeltaFromEnv,
  type TieBreakSignals,
} from "./fusion";
import { hydrateProducts } from "./d1";

export type { Env } from "./types";

const RESULT_LIMIT = 50;

export default {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/health" || url.pathname === "/healthz") {
      return Response.json({
        status: "ok",
        image_search_enabled: env.IMAGE_SEARCH_ENABLED === "true",
        hybrid_fusion_enabled: env.HYBRID_FUSION_ENABLED === "true",
      });
    }

    if (url.pathname === "/search" && request.method === "POST") {
      return handleSearch(request, env, url);
    }

    return new Response("Not Found", { status: 404 });
  },
} satisfies ExportedHandler<Env>;

/** Extract a user-supplied text query (query param `q` or `X-Search-Text`). */
function requestTextQuery(request: Request, url: URL): string {
  return (
    url.searchParams.get("q") ||
    request.headers.get("x-search-text") ||
    ""
  ).trim();
}

async function handleSearch(request: Request, env: Env, url: URL): Promise<Response> {
  // 1. Rate limit.
  const allowed = await checkRateLimit(env, request);
  if (!allowed) {
    return Response.json(
      { error: "rate_limited", detail: "Too many requests." },
      { status: 429 },
    );
  }

  const imageSearchEnabled = env.IMAGE_SEARCH_ENABLED === "true";
  const hybridEnabled = env.HYBRID_FUSION_ENABLED === "true";

  // --- Pure text-search path (flags off): ZERO ML dependency. ---------------
  if (!imageSearchEnabled) {
    return handleTextOnly(request, env, url, hybridEnabled);
  }

  const fusionParams = fusionParamsFromEnv(env);

  // 2. Read the uploaded photo ONCE (strict max-size). Reused for both ML calls.
  let photo: ArrayBuffer;
  try {
    photo = await readUploadOnce(request, env);
  } catch (err) {
    return uploadErrorResponse(err);
  }

  // 3. Fan out the SAME bytes to /search and /ocr in parallel.
  const searchPromise: Promise<MlSearchResponse> = mlClient
    .search(env, photo)
    .catch((err) => {
      // Auth failure must fail closed for the ML tier -> surfaced below.
      if (err instanceof MlAuthError) throw err;
      return { products: [], weak_visual_match: true } as MlSearchResponse;
    });

  const ocrPromise: Promise<MlOcrResponse> = mlClient
    .ocr(env, photo)
    .catch((err) => {
      if (err instanceof MlAuthError) throw err;
      return { candidates: [], tokens: [] } as MlOcrResponse;
    });

  // Opt-in ImageKit persistence (default: no write).
  const persistPromise = maybePersistToImageKit(env, photo, "query.jpg").catch(
    () => ({ persisted: false as const }),
  );

  let mlSearch: MlSearchResponse;
  let mlOcr: MlOcrResponse;
  try {
    [mlSearch, mlOcr] = await Promise.all([searchPromise, ocrPromise]);
  } catch (err) {
    if (err instanceof MlAuthError) {
      // Fail closed: never serve when ML auth cannot be satisfied.
      return Response.json(
        { error: "ml_auth_failed", detail: "ML service authentication is not configured." },
        { status: 502 },
      );
    }
    return Response.json(
      { error: "upstream_error", detail: "ML service call failed." },
      { status: 502 },
    );
  }
  await persistPromise;

  // 4. Text search from OCR candidates (graceful degradation to []).
  const ocrQuery = mlOcr.candidates.join(" ").trim();
  const textList: RankedItem[] = await textAdapter.search(env, ocrQuery, RESULT_LIMIT);

  // 5. Fuse.
  const imageList = rankMlHits(mlSearch.products);
  let fused: FusedItem[];
  if (!hybridEnabled) {
    // Image-only (fusion disabled).
    fused = imageList.map((i) => ({ product_id: i.product_id, score: i.score }));
  } else {
    fused = fuse(imageList, textList, mlSearch.weak_visual_match, fusionParams);
  }

  fused = fused.slice(0, RESULT_LIMIT);

  // 6. Hydrate metadata (D1) + 8. batch price hydrate — in parallel.
  const productIds = fused.map((f) => f.product_id);
  const [metaMap, priceMap] = await Promise.all([
    hydrateProducts(env.DB, productIds).catch(
      () => new Map<string, ProductMetadata>(),
    ),
    commerce.getPrices(env, productIds).catch((): PriceMap => ({})),
  ]);

  // 7. Near-tie tie-break using OCR-derived name/strength + manufacturer +
  //    barcode. Only products present in D1 participate; order is otherwise
  //    the fusion order.
  const ordered = tieBreakOrder(fused, metaMap, ocrQuery, tieBreakDeltaFromEnv(env));

  const products = shapeProducts(ordered, metaMap, priceMap);
  return Response.json(buildBody(products, mlOcr.candidates, mlSearch.weak_visual_match, {
    mode: !hybridEnabled ? "image-only" : fusionParams.mode,
    hybridEnabled,
    imageSearchEnabled: true,
  }));
}

/**
 * Pure text-search path: no ML calls at all. Query comes from the request.
 * Works even if ML is down/unconfigured (additive rollback).
 */
async function handleTextOnly(
  request: Request,
  env: Env,
  url: URL,
  hybridEnabled: boolean,
): Promise<Response> {
  const query = requestTextQuery(request, url);
  const textList = query
    ? await textAdapter.search(env, query, RESULT_LIMIT)
    : [];

  const fused: FusedItem[] = textList
    .slice(0, RESULT_LIMIT)
    .map((t) => ({ product_id: t.product_id, score: t.score }));
  const productIds = fused.map((f) => f.product_id);

  const [metaMap, priceMap] = await Promise.all([
    hydrateProducts(env.DB, productIds).catch(
      () => new Map<string, ProductMetadata>(),
    ),
    commerce.getPrices(env, productIds).catch((): PriceMap => ({})),
  ]);

  const products = shapeProducts(fused, metaMap, priceMap);
  return Response.json(buildBody(products, [], false, {
    mode: "text-only",
    hybridEnabled,
    imageSearchEnabled: false,
  }));
}

/** Reorder fused items within near-tie clusters using OCR signals. */
function tieBreakOrder(
  fused: FusedItem[],
  metaMap: Map<string, ProductMetadata>,
  ocrText: string,
  delta: number,
): FusedItem[] {
  if (!ocrText.trim()) return fused;
  const signals: TieBreakSignals[] = [];
  for (const f of fused) {
    const meta = metaMap.get(f.product_id);
    if (!meta) continue;
    signals.push({
      product_id: f.product_id,
      score: f.score,
      name: meta.name,
      strength: meta.strength,
      manufacturer: meta.manufacturer,
      barcode: meta.barcode,
    });
  }
  const reordered = applyTieBreak(signals, ocrText, delta);
  return reordered.map((s) => ({ product_id: s.product_id, score: s.score }));
}

/** Shape fused items into the API product list (only products present in D1). */
function shapeProducts(
  fused: FusedItem[],
  metaMap: Map<string, ProductMetadata>,
  priceMap: PriceMap,
): ShapedProduct[] {
  const products: ShapedProduct[] = [];
  for (const f of fused) {
    const meta = metaMap.get(f.product_id);
    if (!meta) continue;
    const price = priceMap[f.product_id];
    products.push({
      product_id: f.product_id,
      score: f.score,
      name: meta.name,
      manufacturer: meta.manufacturer,
      strength: meta.strength,
      sku: meta.sku,
      barcode: meta.barcode,
      image_url: meta.imagekit_url,
      price: price ? price.price : null,
      currency: price ? price.currency : null,
      in_stock: price ? price.in_stock : null,
    });
  }
  return products;
}

function buildBody(
  products: ShapedProduct[],
  ocrCandidates: string[],
  weakVisualMatch: boolean,
  fusion: { mode: string; hybridEnabled: boolean; imageSearchEnabled: boolean },
): SearchApiResponse {
  return {
    products,
    ocr_candidates: ocrCandidates,
    weak_visual_match: weakVisualMatch,
    fusion: {
      mode: fusion.mode,
      hybrid_enabled: fusion.hybridEnabled,
      image_search_enabled: fusion.imageSearchEnabled,
    },
  };
}

function uploadErrorResponse(err: unknown): Response {
  if (err instanceof UploadTooLargeError) {
    return Response.json(
      { error: "payload_too_large", detail: err.message },
      { status: 413 },
    );
  }
  if (err instanceof EmptyUploadError) {
    return Response.json(
      { error: "bad_request", detail: err.message },
      { status: 400 },
    );
  }
  return Response.json(
    { error: "bad_request", detail: "Could not read upload." },
    { status: 400 },
  );
}
