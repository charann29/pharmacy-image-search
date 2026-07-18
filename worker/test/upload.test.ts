import { describe, it, expect } from "vitest";
import {
  readUploadOnce,
  maybePersistToImageKit,
  persistenceEnabled,
  maxUploadBytes,
  UploadTooLargeError,
  EmptyUploadError,
  DEFAULT_MAX_UPLOAD_BYTES,
} from "../src/upload";
import type { Env } from "../src/types";

function env(overrides: Partial<Env> = {}): Env {
  return { ...overrides } as Env;
}

function req(body: BodyInit | null, headers: Record<string, string> = {}): Request {
  return new Request("https://w/search", { method: "POST", body, headers });
}

describe("upload — read once + max size", () => {
  it("reads bytes once into an ArrayBuffer", async () => {
    const bytes = new Uint8Array([1, 2, 3, 4]);
    const buf = await readUploadOnce(req(bytes), env());
    expect(buf.byteLength).toBe(4);
  });

  it("uses the default max when unset", () => {
    expect(maxUploadBytes(env())).toBe(DEFAULT_MAX_UPLOAD_BYTES);
  });

  it("rejects an over-limit upload by Content-Length", async () => {
    await expect(
      readUploadOnce(
        req(new Uint8Array([1, 2, 3]), { "content-length": "999999" }),
        env({ MAX_UPLOAD_BYTES: "10" }),
      ),
    ).rejects.toBeInstanceOf(UploadTooLargeError);
  });

  it("rejects an over-limit upload by buffered size", async () => {
    const big = new Uint8Array(50);
    await expect(
      readUploadOnce(req(big), env({ MAX_UPLOAD_BYTES: "10" })),
    ).rejects.toBeInstanceOf(UploadTooLargeError);
  });

  it("rejects an empty upload", async () => {
    await expect(
      readUploadOnce(req(new Uint8Array([])), env()),
    ).rejects.toBeInstanceOf(EmptyUploadError);
  });
});

describe("upload — persistence default off", () => {
  it("does not persist by default (no ImageKit write)", async () => {
    expect(persistenceEnabled(env())).toBe(false);
    const res = await maybePersistToImageKit(env(), new ArrayBuffer(4), "q.jpg");
    expect(res.persisted).toBe(false);
  });

  it("no-ops when persistence flag on but private key missing", async () => {
    const res = await maybePersistToImageKit(
      env({ PERSIST_QUERY_PHOTOS: "true" }),
      new ArrayBuffer(4),
      "q.jpg",
    );
    expect(res.persisted).toBe(false);
  });
});
