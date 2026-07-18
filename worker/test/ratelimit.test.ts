import { describe, it, expect } from "vitest";
import { checkRateLimit, rateLimitKey } from "../src/ratelimit";
import type { Env, RateLimitBinding } from "../src/types";

function req(headers: Record<string, string> = {}): Request {
  return new Request("https://w/search", { method: "POST", headers });
}

/** Stub binding that allows `limit` calls then blocks. */
function limiter(allowFor: number): RateLimitBinding {
  let calls = 0;
  return {
    async limit() {
      calls++;
      return { success: calls <= allowFor };
    },
  };
}

describe("ratelimit key derivation", () => {
  it("prefers token over IP", () => {
    const k = rateLimitKey(req({ authorization: "Bearer abc" }));
    expect(k.startsWith("tok:")).toBe(true);
  });
  it("falls back to CF-Connecting-IP", () => {
    const k = rateLimitKey(req({ "cf-connecting-ip": "1.2.3.4" }));
    expect(k).toBe("ip:1.2.3.4");
  });
});

describe("checkRateLimit", () => {
  it("allows under limit, blocks over limit", async () => {
    const env = { RATE_LIMITER: limiter(2) } as unknown as Env;
    const r = req({ "cf-connecting-ip": "9.9.9.9" });
    expect(await checkRateLimit(env, r)).toBe(true);
    expect(await checkRateLimit(env, r)).toBe(true);
    expect(await checkRateLimit(env, r)).toBe(false); // over limit
  });

  it("fails open when binding is absent", async () => {
    const env = {} as Env;
    expect(await checkRateLimit(env, req())).toBe(true);
  });
});
