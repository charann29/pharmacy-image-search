import { describe, it, expect, beforeEach } from "vitest";
import { fetchMock } from "cloudflare:test";
import * as textAdapter from "../src/text_search_adapter";
import * as commerce from "../src/commerce_adapter";
import type { Env } from "../src/types";

const TEXT_URL = "https://text.example.com";
const COMMERCE_URL = "https://commerce.example.com";

beforeEach(() => {
  fetchMock.activate();
  fetchMock.disableNetConnect();
});

function env(overrides: Partial<Env> = {}): Env {
  return {
    TEXT_SEARCH_URL: `${TEXT_URL}/search`,
    TEXT_SEARCH_AUTH: "Bearer text-secret",
    COMMERCE_API_URL: `${COMMERCE_URL}/prices`,
    COMMERCE_API_AUTH: "Bearer commerce-secret",
    ...overrides,
  } as Env;
}

describe("text_search_adapter", () => {
  it("maps { results: [...] } to ranked items", async () => {
    fetchMock
      .get(TEXT_URL)
      .intercept({ path: "/search", method: "POST" })
      .reply(
        200,
        JSON.stringify({
          results: [
            { product_id: "A", score: 0.9 },
            { product_id: "B", score: 0.5 },
          ],
        }),
      );
    const out = await textAdapter.search(env(), "crocin 650");
    expect(out).toEqual([
      { product_id: "A", score: 0.9, rank: 1 },
      { product_id: "B", score: 0.5, rank: 2 },
    ]);
  });

  it("accepts a bare array response", async () => {
    fetchMock
      .get(TEXT_URL)
      .intercept({ path: "/search", method: "POST" })
      .reply(200, JSON.stringify([{ product_id: "Z", score: 1 }]));
    const out = await textAdapter.search(env(), "q");
    expect(out).toEqual([{ product_id: "Z", score: 1, rank: 1 }]);
  });

  it("returns [] when TEXT_SEARCH_URL is unset (graceful)", async () => {
    const out = await textAdapter.search(env({ TEXT_SEARCH_URL: "" }), "q");
    expect(out).toEqual([]);
  });

  it("returns [] on empty query", async () => {
    const out = await textAdapter.search(env(), "   ");
    expect(out).toEqual([]);
  });

  it("degrades to [] on non-2xx", async () => {
    fetchMock
      .get(TEXT_URL)
      .intercept({ path: "/search", method: "POST" })
      .reply(500, "boom");
    const out = await textAdapter.search(env(), "q");
    expect(out).toEqual([]);
  });

  it("degrades to [] on timeout", async () => {
    fetchMock
      .get(TEXT_URL)
      .intercept({ path: "/search", method: "POST" })
      .reply(200, JSON.stringify({ results: [] }))
      .delay(300);
    const out = await textAdapter.search(
      env({ TEXT_SEARCH_TIMEOUT_MS: "20" }),
      "q",
    );
    expect(out).toEqual([]);
  });
});

describe("commerce_adapter", () => {
  it("batch-fetches and maps nested { prices } form", async () => {
    let capturedBody = "";
    fetchMock
      .get(COMMERCE_URL)
      .intercept({ path: "/prices", method: "POST" })
      .reply(200, (opts: any) => {
        capturedBody = String(opts.body);
        return JSON.stringify({
          prices: {
            A: { price: 12.5, currency: "INR", in_stock: true },
            B: { price: 4, currency: "INR", in_stock: false },
          },
        });
      });
    const out = await commerce.getPrices(env(), ["A", "B", "A"]);
    expect(out).toEqual({
      A: { price: 12.5, currency: "INR", in_stock: true },
      B: { price: 4, currency: "INR", in_stock: false },
    });
    // Single batched request with deduped ids.
    expect(JSON.parse(capturedBody)).toEqual({ product_ids: ["A", "B"] });
  });

  it("accepts array form", async () => {
    fetchMock
      .get(COMMERCE_URL)
      .intercept({ path: "/prices", method: "POST" })
      .reply(
        200,
        JSON.stringify([
          { product_id: "A", price: 9, currency: "USD", in_stock: true },
        ]),
      );
    const out = await commerce.getPrices(env(), ["A"]);
    expect(out.A).toEqual({ price: 9, currency: "USD", in_stock: true });
  });

  it("returns {} when unconfigured (graceful)", async () => {
    const out = await commerce.getPrices(env({ COMMERCE_API_URL: "" }), ["A"]);
    expect(out).toEqual({});
  });

  it("degrades to {} on timeout (price omitted, not failing)", async () => {
    fetchMock
      .get(COMMERCE_URL)
      .intercept({ path: "/prices", method: "POST" })
      .reply(200, JSON.stringify({ prices: {} }))
      .delay(300);
    const out = await commerce.getPrices(
      env({ COMMERCE_TIMEOUT_MS: "20" }),
      ["A"],
    );
    expect(out).toEqual({});
  });

  it("degrades to {} on non-2xx", async () => {
    fetchMock
      .get(COMMERCE_URL)
      .intercept({ path: "/prices", method: "POST" })
      .reply(503, "down");
    const out = await commerce.getPrices(env(), ["A"]);
    expect(out).toEqual({});
  });
});
