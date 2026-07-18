/**
 * ml_client.ts — hardened authenticated transport to the self-hosted ML
 * service (`ML_SERVICE_URL`).
 *
 * ## Auth mechanism (v1)
 * We implement **HMAC-signed requests** to mirror `ml-service/app/auth.py`
 * exactly. For each request we send:
 *
 *   - `X-Timestamp`: unix seconds (int, as a string)
 *   - `X-Signature`: hex(HMAC_SHA256(key, `${timestamp}.${METHOD}.${path}`))
 *
 * The signature is **path-bound** and **timestamped**; the service rejects
 * signatures whose timestamp is outside its replay window (default 300s) or
 * whose signature does not match — so a captured request cannot be replayed
 * against a different path or after the window closes.
 *
 * The signing key is `ML_SERVICE_SHARED_SECRET`, read from the environment
 * (never written literally into source). If it is missing we **fail closed**:
 * no unauthenticated request is ever sent.
 *
 * ### Alternative: Cloudflare Tunnel + Access service token
 * In production the ML service can instead sit behind a Cloudflare Tunnel with
 * Access enforced; the Worker then attaches `CF-Access-Client-Id` /
 * `CF-Access-Client-Secret` service-token headers and Access terminates auth
 * at the edge. That removes app-level signing entirely. We ship HMAC as the
 * concrete default because it matches `auth.py`; the Tunnel/Access option is
 * documented in docs/runbook.md and toggled purely by deployment topology.
 */

import type { Env, MlSearchResponse, MlOcrResponse } from "./types";
import { envNumber } from "./util";

const DEFAULT_TIMEOUT_MS = 8000;

/** Raised when auth cannot be satisfied — request is never sent (fail closed). */
export class MlAuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MlAuthError";
  }
}

/** Raised for transport/HTTP failures talking to the ML service. */
export class MlServiceError extends Error {
  readonly status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "MlServiceError";
    this.status = status;
  }
}

function toHex(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let out = "";
  for (let i = 0; i < bytes.length; i++) {
    out += bytes[i].toString(16).padStart(2, "0");
  }
  return out;
}

/**
 * Compute the HMAC-SHA256 signature over `${timestamp}.${METHOD}.${path}`,
 * matching `ml-service/app/auth.py::compute_signature`.
 */
export async function computeSignature(
  key: string,
  timestamp: string,
  method: string,
  path: string,
): Promise<string> {
  const enc = new TextEncoder();
  const cryptoKey = await crypto.subtle.importKey(
    "raw",
    enc.encode(key),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const message = enc.encode(`${timestamp}.${method.toUpperCase()}.${path}`);
  const sig = await crypto.subtle.sign("HMAC", cryptoKey, message);
  return toHex(sig);
}

function timeoutMs(env: Env): number {
  return envNumber(env.ML_TIMEOUT_MS, DEFAULT_TIMEOUT_MS, { positive: true });
}

/**
 * Perform an authenticated POST to the ML service. Fails closed when the
 * signing key is absent. Bounded by a timeout.
 */
async function authedPost(
  env: Env,
  path: string,
  body: ArrayBuffer,
  nowSeconds?: number,
): Promise<Response> {
  const key = env.ML_SERVICE_SHARED_SECRET;
  if (!key) {
    // Fail closed: never emit an unauthenticated request.
    throw new MlAuthError(
      "ML_SERVICE_SHARED_SECRET is not configured; refusing to call ML service unauthenticated.",
    );
  }

  const base = (env.ML_SERVICE_URL || "").replace(/\/+$/, "");
  if (!base) {
    throw new MlServiceError("ML_SERVICE_URL is not configured.");
  }

  const timestamp = String(
    Math.floor(nowSeconds !== undefined ? nowSeconds : Date.now() / 1000),
  );
  const signature = await computeSignature(key, timestamp, "POST", path);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs(env));
  try {
    const resp = await fetch(`${base}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Timestamp": timestamp,
        "X-Signature": signature,
      },
      body,
      signal: controller.signal,
    });
    if (!resp.ok) {
      throw new MlServiceError(
        `ML service ${path} returned ${resp.status}`,
        resp.status,
      );
    }
    return resp;
  } catch (err) {
    if (err instanceof MlServiceError || err instanceof MlAuthError) throw err;
    throw new MlServiceError(
      `ML service ${path} request failed: ${(err as Error).message}`,
    );
  } finally {
    clearTimeout(timer);
  }
}

/**
 * POST image bytes to ML `/search`.
 * Production query flow uses `/search` + `/ocr` only (NOT `/embed`).
 */
export async function search(
  env: Env,
  imageBytes: ArrayBuffer,
): Promise<MlSearchResponse> {
  const resp = await authedPost(env, "/search", imageBytes);
  const json = (await resp.json()) as Partial<MlSearchResponse>;
  return {
    products: Array.isArray(json.products) ? json.products : [],
    weak_visual_match: Boolean(json.weak_visual_match),
  };
}

/** POST image bytes to ML `/ocr`. */
export async function ocr(
  env: Env,
  imageBytes: ArrayBuffer,
): Promise<MlOcrResponse> {
  const resp = await authedPost(env, "/ocr", imageBytes);
  const json = (await resp.json()) as Partial<MlOcrResponse>;
  return {
    candidates: Array.isArray(json.candidates) ? json.candidates : [],
    tokens: Array.isArray(json.tokens) ? json.tokens : [],
  };
}
