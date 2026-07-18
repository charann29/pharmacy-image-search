/**
 * index.ts — image-search Worker router / orchestration.
 *
 * Image-search endpoint flow (POST /search):
 *   1. rate-limit (per IP/token) via the Workers binding
 *   2. read the uploaded photo ONCE into an ArrayBuffer under a strict max-size
 *      limit (a request body stream cannot be consumed twice)
 *   3. send the SAME bytes to ML `/search` and `/ocr` in parallel
 *   4. forward OCR text to the text-search adapter
 *   5. fuse image + text results (RRF default / weighted; weak-match aware)
 *   6. hydrate product metadata from D1 (binding)
 *   7. batch-hydrate price/availability from the commerce adapter
 *   8. return a shaped JSON response
 *
 * Feature flags:
 *   - IMAGE_SEARCH_ENABLED  = "false" -> pure text-search behavior (no ML calls)
 *   - HYBRID_FUSION_ENABLED = "false" -> no fusion; use image-only (or text-only
 *     when image search is disabled)
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
import { fuse, fusionParamsFromEnv, rankMlHits } from "./fusion";
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
      return handleSearch(request, env);
    }

    return new Response("Not Found", { status: 404 });
  },
} satisfies ExportedHandler<Env>;

async function handleSearch(request: Request, env: Env): Promise<Response> {
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
  const fusionParams = fusionParamsFromEnv(env);

  // 2. Read the uploaded photo ONCE (strict max-size). Even in text-only mode
  //    we need it for OCR-derived query text unless a text query is supplied.
  let photo: ArrayBuffer;
  try {
    photo = await readUploadOnce(request, env);
  } catch (err) {
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

  // 3. Fan out. When image search is disabled we skip the ML /search call but
  //    still run OCR to drive the text search (pure text-search behavior).
  const searchPromise: Promise<MlSearchResponse> = imageSearchEnabled
    ? mlClient.search(env, photo).catch((err) => {
        // Auth failure must fail closed for the ML tier -> surfaced below.
        if (err instanceof MlAuthError) throw err;
        return { products: [], weak_visual_match: true } as MlSearchResponse;
      })
    : Promise.resolve({ products: [], weak_visual_match: true } as MlSearchResponse);

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
  if (!imageSearchEnabled) {
    // Pure text-search behavior.
    fused = textList.map((t) => ({ product_id: t.product_id, score: t.score }));
  } else if (!hybridEnabled) {
    // Image-only (fusion disabled).
    fused = imageList.map((i) => ({ product_id: i.product_id, score: i.score }));
  } else {
    fused = fuse(imageList, textList, mlSearch.weak_visual_match, fusionParams);
  }

  fused = fused.slice(0, RESULT_LIMIT);
  const productIds = fused.map((f) => f.product_id);

  // 6. Hydrate metadata from D1 (binding) + 7. batch price hydrate — parallel.
  const [metaMap, priceMap] = await Promise.all([
    hydrateProducts(env.DB, productIds).catch(
      () => new Map<string, ProductMetadata>(),
    ),
    commerce.getPrices(env, productIds).catch((): PriceMap => ({})),
  ]);

  // 8. Shape response (only include products present in D1).
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

  const body: SearchApiResponse = {
    products,
    ocr_candidates: mlOcr.candidates,
    weak_visual_match: mlSearch.weak_visual_match,
    fusion: {
      mode: !imageSearchEnabled
        ? "text-only"
        : !hybridEnabled
          ? "image-only"
          : fusionParams.mode,
      hybrid_enabled: hybridEnabled,
      image_search_enabled: imageSearchEnabled,
    },
  };
  return Response.json(body);
}
