/**
 * ratelimit.ts — per-IP / per-token limiting via the Workers rate-limiting
 * binding (`env.RATE_LIMITER`, declared in wrangler.toml).
 *
 * We derive a stable key preferring an authenticated token (Authorization /
 * X-Api-Token header) over the client IP (`CF-Connecting-IP`), so that a
 * signed-in caller is limited by identity and anonymous callers by IP.
 */

import type { Env } from "./types";

/** Derive the rate-limit key for a request (token if present, else IP). */
export function rateLimitKey(request: Request): string {
  const auth = request.headers.get("authorization");
  if (auth) {
    // Bucket by a short, non-reversible fingerprint of the token, not the raw
    // secret, so the key is stable without logging credentials.
    return `tok:${fingerprint(auth)}`;
  }
  const apiToken = request.headers.get("x-api-token");
  if (apiToken) {
    return `tok:${fingerprint(apiToken)}`;
  }
  const ip =
    request.headers.get("cf-connecting-ip") ||
    request.headers.get("x-forwarded-for") ||
    "unknown";
  return `ip:${ip}`;
}

/** Small deterministic non-cryptographic fingerprint (djb2). */
function fingerprint(input: string): string {
  let hash = 5381;
  for (let i = 0; i < input.length; i++) {
    hash = ((hash << 5) + hash + input.charCodeAt(i)) | 0;
  }
  return (hash >>> 0).toString(36);
}

/**
 * Returns true when the request is allowed, false when it should be blocked.
 * If the binding is unavailable (e.g. misconfigured local env), fail open so
 * the health/other paths still work — the binding is the enforcement point in
 * production.
 */
export async function checkRateLimit(
  env: Env,
  request: Request,
): Promise<boolean> {
  if (!env.RATE_LIMITER || typeof env.RATE_LIMITER.limit !== "function") {
    return true;
  }
  const { success } = await env.RATE_LIMITER.limit({
    key: rateLimitKey(request),
  });
  return success;
}
