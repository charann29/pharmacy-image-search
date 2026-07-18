// Vitest + @cloudflare/vitest-pool-workers smoke test for the Worker scaffold.
// Task 8a/8b add binding-level tests (D1 join, rate limiter, ml_client auth).
import { env, createExecutionContext, waitOnExecutionContext } from "cloudflare:test";
import { describe, it, expect } from "vitest";
import worker from "../src/index";

describe("worker health", () => {
  it("responds 200 on /health", async () => {
    const request = new Request("http://example.com/health");
    const ctx = createExecutionContext();
    const response = await worker.fetch(request, env as any, ctx);
    await waitOnExecutionContext(ctx);
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body).toHaveProperty("status", "ok");
  });
});
