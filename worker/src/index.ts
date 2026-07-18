/**
 * Image-search Worker — router / orchestration entrypoint.
 *
 * This scaffold wires a minimal router with a working health path so
 * `wrangler deploy --dry-run` succeeds. Orchestration logic (upload fan-out,
 * ml_client, fusion, D1 hydrate, commerce/text adapters) is filled in by
 * Tasks 8a/8b — see sibling modules referenced in the plan.
 */

export interface Env {
  // D1 binding (see wrangler.toml). Bound at runtime.
  DB: D1Database;
  // Workers rate-limiting binding.
  RATE_LIMITER: RateLimit;
  // Non-secret config vars.
  ML_SERVICE_URL: string;
  TEXT_SEARCH_URL: string;
  COMMERCE_API_URL: string;
  IMAGE_SEARCH_ENABLED: string;
  HYBRID_FUSION_ENABLED: string;
  // Secrets (set via `wrangler secret put`, referenced by name only).
  IMAGEKIT_PUBLIC_KEY?: string;
  IMAGEKIT_PRIVATE_KEY?: string;
  ML_SERVICE_SHARED_SECRET?: string;
  TEXT_SEARCH_AUTH?: string;
  COMMERCE_API_AUTH?: string;
}

// Minimal RateLimit binding type (Workers rate-limiting binding).
interface RateLimit {
  limit(options: { key: string }): Promise<{ success: boolean }>;
}

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
      // STUB (Task 8b): read photo once -> fan out to /search + /ocr ->
      // text adapter -> fusion -> D1 hydrate -> commerce prices -> shaped JSON.
      return Response.json(
        { status: "not_implemented", detail: "/search orchestration lands in Task 8b." },
        { status: 501 },
      );
    }

    return new Response("Not Found", { status: 404 });
  },
} satisfies ExportedHandler<Env>;
