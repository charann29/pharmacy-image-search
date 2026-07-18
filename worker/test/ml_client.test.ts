import { describe, it, expect, beforeEach } from "vitest";
import { fetchMock } from "cloudflare:test";
import * as mlClient from "../src/ml_client";
import { MlAuthError, computeSignature } from "../src/ml_client";
import type { Env } from "../src/types";

const BASE = "https://ml.example.com";

function makeEnv(overrides: Partial<Env> = {}): Env {
  return {
    ML_SERVICE_URL: BASE,
    ML_SERVICE_SHARED_SECRET: "test-signing-key",
    ...overrides,
  } as Env;
}

beforeEach(() => {
  fetchMock.activate();
  fetchMock.disableNetConnect();
});

describe("ml_client auth (fails closed)", () => {
  it("throws MlAuthError and issues no request when signing key is missing", async () => {
    const env = makeEnv({ ML_SERVICE_SHARED_SECRET: undefined });
    await expect(mlClient.search(env, new ArrayBuffer(4))).rejects.toBeInstanceOf(
      MlAuthError,
    );
    await expect(mlClient.ocr(env, new ArrayBuffer(4))).rejects.toBeInstanceOf(
      MlAuthError,
    );
  });

  it("signs /search with a path-bound HMAC + timestamp header", async () => {
    const env = makeEnv();
    let capturedHeaders: Record<string, string> = {};
    fetchMock
      .get(BASE)
      .intercept({ path: "/search", method: "POST" })
      .reply(200, (opts: any) => {
        capturedHeaders = normalizeHeaders(opts.headers);
        return JSON.stringify({
          products: [{ product_id: "A", score: 0.9 }],
          weak_visual_match: false,
        });
      });

    const res = await mlClient.search(env, new ArrayBuffer(8));
    expect(res.products[0].product_id).toBe("A");
    expect(capturedHeaders["x-timestamp"]).toBeTruthy();
    expect(capturedHeaders["x-signature"]).toBeTruthy();

    // Signature must match the auth.py scheme: HMAC over `${ts}.POST./search`.
    const ts = capturedHeaders["x-timestamp"];
    const expected = await computeSignature("test-signing-key", ts, "POST", "/search");
    expect(capturedHeaders["x-signature"]).toBe(expected);
  });

  it("binds signature to the path (/ocr differs from /search)", async () => {
    const ts = "1700000000";
    const sigSearch = await computeSignature("k", ts, "POST", "/search");
    const sigOcr = await computeSignature("k", ts, "POST", "/ocr");
    expect(sigSearch).not.toBe(sigOcr);
  });

  it("raises MlServiceError on non-2xx", async () => {
    const env = makeEnv();
    fetchMock
      .get(BASE)
      .intercept({ path: "/ocr", method: "POST" })
      .reply(401, "unauthorized");
    await expect(mlClient.ocr(env, new ArrayBuffer(4))).rejects.toMatchObject({
      name: "MlServiceError",
      status: 401,
    });
  });
});

function normalizeHeaders(headers: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (!headers) return out;
  if (Array.isArray(headers)) {
    for (let i = 0; i < headers.length; i += 2) {
      out[String(headers[i]).toLowerCase()] = String(headers[i + 1]);
    }
  } else if (typeof headers === "object") {
    for (const [k, v] of Object.entries(headers as Record<string, unknown>)) {
      out[k.toLowerCase()] = String(v);
    }
  }
  return out;
}
