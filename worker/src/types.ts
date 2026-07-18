/**
 * Shared types for the image-search Worker.
 *
 * `Env` mirrors the bindings + vars declared in wrangler.toml. Secrets are
 * referenced by NAME only (set via `wrangler secret put <NAME>`); literal
 * values never live in source.
 */

/** Minimal Workers rate-limiting binding surface we depend on. */
export interface RateLimitBinding {
  limit(options: { key: string }): Promise<{ success: boolean }>;
}

export interface Env {
  // --- Bindings -----------------------------------------------------------
  /** D1 binding (see wrangler.toml). Bound at runtime. */
  DB: D1Database;
  /** Workers rate-limiting binding. */
  RATE_LIMITER: RateLimitBinding;

  // --- Non-secret config vars --------------------------------------------
  ML_SERVICE_URL: string;
  TEXT_SEARCH_URL: string;
  COMMERCE_API_URL: string;
  IMAGE_SEARCH_ENABLED: string;
  HYBRID_FUSION_ENABLED: string;

  // Optional tuning vars (fall back to sane defaults when unset/empty).
  /** Fusion mode: "rrf" (default) or "weighted". */
  FUSION_MODE?: string;
  W_IMAGE?: string;
  W_TEXT?: string;
  RRF_K?: string;
  /** Multiplier applied to image weight when weak_visual_match is set. */
  WEAK_VISUAL_IMAGE_MULTIPLIER?: string;
  /** Strict max upload size in bytes. */
  MAX_UPLOAD_BYTES?: string;
  /** Per-request timeout (ms) for ML/text/commerce calls. */
  ML_TIMEOUT_MS?: string;
  TEXT_SEARCH_TIMEOUT_MS?: string;
  COMMERCE_TIMEOUT_MS?: string;
  /** "true" to persist query photos to ImageKit (opt-in, off by default). */
  PERSIST_QUERY_PHOTOS?: string;
  IMAGEKIT_UPLOAD_URL?: string;

  // --- Secrets (set via `wrangler secret put`, referenced by NAME only) ---
  IMAGEKIT_PUBLIC_KEY?: string;
  IMAGEKIT_PRIVATE_KEY?: string;
  /** Shared secret / HMAC signing key for Worker->ML-service auth. */
  ML_SERVICE_SHARED_SECRET?: string;
  TEXT_SEARCH_AUTH?: string;
  COMMERCE_API_AUTH?: string;
}

// --- ML service contract ---------------------------------------------------

/** One product hit from the ML `/search` endpoint. */
export interface MlProductHit {
  product_id: string;
  score: number;
}

/** Response body of ML `POST /search`. */
export interface MlSearchResponse {
  products: MlProductHit[];
  weak_visual_match: boolean;
}

/** Response body of ML `POST /ocr`. */
export interface MlOcrResponse {
  candidates: string[];
  tokens: unknown[];
}

// --- Ranking / fusion ------------------------------------------------------

/** A ranked candidate used by adapters + fusion. */
export interface RankedItem {
  product_id: string;
  score: number;
  rank: number;
}

/** Fused result before hydration. */
export interface FusedItem {
  product_id: string;
  score: number;
}

// --- D1 hydration ----------------------------------------------------------

/** Product metadata hydrated from D1. */
export interface ProductMetadata {
  product_id: string;
  sku: string | null;
  name: string | null;
  manufacturer: string | null;
  strength: string | null;
  barcode: string | null;
  imagekit_url: string | null;
}

// --- Commerce --------------------------------------------------------------

export interface PriceInfo {
  price: number | null;
  currency: string | null;
  in_stock: boolean;
}

export type PriceMap = Record<string, PriceInfo>;

// --- Shaped API response ---------------------------------------------------

export interface ShapedProduct {
  product_id: string;
  score: number;
  name: string | null;
  manufacturer: string | null;
  strength: string | null;
  sku: string | null;
  barcode: string | null;
  image_url: string | null;
  price: number | null;
  currency: string | null;
  in_stock: boolean | null;
}

export interface SearchApiResponse {
  products: ShapedProduct[];
  ocr_candidates: string[];
  weak_visual_match: boolean;
  fusion: {
    mode: string;
    hybrid_enabled: boolean;
    image_search_enabled: boolean;
  };
}
