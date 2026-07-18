/**
 * upload.ts — query-photo intake.
 *
 * ## Privacy default (do NOT persist query photos)
 * The default flow reads the uploaded photo bytes ONCE into an `ArrayBuffer`
 * and forwards them to the ML service. Query photos are **not persisted** —
 * they can contain PII (faces, labels, personal data) and persisting them adds
 * storage cost and compliance burden.
 *
 * ## Opt-in ImageKit persistence
 * Persisting is opt-in behind `PERSIST_QUERY_PHOTOS === "true"` with documented
 * retention/consent rules (see docs/runbook.md). When enabled we upload with
 * `Authorization: Basic base64("${IMAGEKIT_PRIVATE_KEY}:")` — note the TRAILING
 * COLON (empty password). The private key is server-side only; it is never
 * exposed to the client.
 *
 * ## Strict max size
 * We enforce a strict max upload size to bound memory + abuse. Requests over
 * the limit are rejected before the body is buffered when a Content-Length is
 * present, and always rejected after reading if the buffered size exceeds it.
 */

import type { Env } from "./types";
import { envNumber } from "./util";

/** Default strict max upload size: 10 MiB. */
export const DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

export class UploadTooLargeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UploadTooLargeError";
  }
}

export class EmptyUploadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EmptyUploadError";
  }
}

export function maxUploadBytes(env: Env): number {
  return envNumber(env.MAX_UPLOAD_BYTES, DEFAULT_MAX_UPLOAD_BYTES, {
    positive: true,
  });
}

/**
 * Read the uploaded photo bytes ONCE under a strict max-size limit.
 *
 * A request body stream cannot be consumed twice, so callers must read here
 * exactly once and reuse the returned `ArrayBuffer` for every downstream call
 * (both `/search` and `/ocr`).
 */
export async function readUploadOnce(
  request: Request,
  env: Env,
): Promise<ArrayBuffer> {
  const limit = maxUploadBytes(env);

  // Fast reject using Content-Length when available.
  const contentLength = request.headers.get("content-length");
  if (contentLength) {
    const declared = Number(contentLength);
    if (Number.isFinite(declared) && declared > limit) {
      throw new UploadTooLargeError(
        `Upload of ${declared} bytes exceeds max ${limit} bytes.`,
      );
    }
  }

  const buffer = await request.arrayBuffer();
  if (buffer.byteLength === 0) {
    throw new EmptyUploadError("Empty upload body.");
  }
  if (buffer.byteLength > limit) {
    throw new UploadTooLargeError(
      `Upload of ${buffer.byteLength} bytes exceeds max ${limit} bytes.`,
    );
  }
  return buffer;
}

/** Whether opt-in ImageKit persistence is enabled. */
export function persistenceEnabled(env: Env): boolean {
  return env.PERSIST_QUERY_PHOTOS === "true";
}

/** base64 of "<privateKey>:" — trailing colon, empty password. */
function imagekitBasicAuth(privateKey: string): string {
  // btoa is available in the Workers runtime.
  return `Basic ${btoa(`${privateKey}:`)}`;
}

export interface ImageKitUploadResult {
  persisted: boolean;
  fileId?: string;
  url?: string;
}

/**
 * Opt-in: persist the query photo to ImageKit. No-ops (returns
 * `{persisted:false}`) unless `PERSIST_QUERY_PHOTOS === "true"`. Server-side
 * only — uses the private key with a trailing-colon Basic credential.
 */
export async function maybePersistToImageKit(
  env: Env,
  bytes: ArrayBuffer,
  fileName: string,
): Promise<ImageKitUploadResult> {
  if (!persistenceEnabled(env)) {
    return { persisted: false };
  }
  const privateKey = env.IMAGEKIT_PRIVATE_KEY;
  if (!privateKey) {
    // Persistence requested but not configured — do not fail the request.
    return { persisted: false };
  }

  const uploadUrl =
    env.IMAGEKIT_UPLOAD_URL || "https://upload.imagekit.io/api/v1/files/upload";

  const form = new FormData();
  form.append("file", new Blob([bytes]), fileName);
  form.append("fileName", fileName);
  // Keep query photos out of the searchable catalog folder.
  form.append("folder", "/query-photos");
  form.append("useUniqueFileName", "true");

  const resp = await fetch(uploadUrl, {
    method: "POST",
    headers: {
      Authorization: imagekitBasicAuth(privateKey),
    },
    body: form,
  });

  if (!resp.ok) {
    // Persistence is best-effort and must never fail the search request.
    return { persisted: false };
  }
  const json = (await resp.json()) as { fileId?: string; url?: string };
  return { persisted: true, fileId: json.fileId, url: json.url };
}
