/**
 * fusion.ts — hybrid fusion of the image-search product list and the text
 * search product list.
 *
 * Two modes:
 *   - **RRF (default):** Reciprocal Rank Fusion. Each list contributes
 *     `weight / (RRF_K + rank)` per product; scores sum across lists. Robust
 *     to differing score scales because it uses rank, not raw score.
 *   - **weighted:** min-max normalize each list's scores to [0,1], then
 *     `W_IMAGE * normImage + W_TEXT * normText`.
 *
 * `weak_visual_match`: when the image tier reports a weak visual match we
 * down-weight the image contribution by `WEAK_VISUAL_IMAGE_MULTIPLIER`
 * (default 0.3) so the ranking leans on text.
 *
 * Tunables (from env vars, with defaults): `FUSION_MODE`, `W_IMAGE`, `W_TEXT`,
 * `RRF_K`, `WEAK_VISUAL_IMAGE_MULTIPLIER`.
 */

import type { Env, MlProductHit, RankedItem, FusedItem } from "./types";
import { envNumber } from "./util";

export interface FusionParams {
  mode: "rrf" | "weighted";
  wImage: number;
  wText: number;
  rrfK: number;
  weakVisualImageMultiplier: number;
}

export const DEFAULT_FUSION_PARAMS: FusionParams = {
  mode: "rrf",
  wImage: 1.0,
  wText: 1.0,
  rrfK: 60,
  weakVisualImageMultiplier: 0.3,
};

function num(value: string | undefined, fallback: number): number {
  return envNumber(value, fallback);
}

/** Resolve fusion params from env vars, falling back to defaults. */
export function fusionParamsFromEnv(env: Env): FusionParams {
  const mode = (env.FUSION_MODE || "").trim().toLowerCase();
  return {
    mode: mode === "weighted" ? "weighted" : "rrf",
    wImage: num(env.W_IMAGE, DEFAULT_FUSION_PARAMS.wImage),
    wText: num(env.W_TEXT, DEFAULT_FUSION_PARAMS.wText),
    rrfK: num(env.RRF_K, DEFAULT_FUSION_PARAMS.rrfK),
    weakVisualImageMultiplier: num(
      env.WEAK_VISUAL_IMAGE_MULTIPLIER,
      DEFAULT_FUSION_PARAMS.weakVisualImageMultiplier,
    ),
  };
}

/** Assign 1-based ranks to a raw ML hit list (already score-ordered). */
export function rankMlHits(hits: MlProductHit[]): RankedItem[] {
  return hits.map((h, i) => ({
    product_id: h.product_id,
    score: h.score,
    rank: i + 1,
  }));
}

/**
 * Fuse image + text ranked lists into a single ordered list.
 *
 * @param imageList image-search results (ranked)
 * @param textList  text-search results (ranked)
 * @param weakVisualMatch when true, image contribution is down-weighted
 * @param params tuning parameters
 */
export function fuse(
  imageList: RankedItem[],
  textList: RankedItem[],
  weakVisualMatch: boolean,
  params: FusionParams = DEFAULT_FUSION_PARAMS,
): FusedItem[] {
  const effectiveImageWeight = weakVisualMatch
    ? params.wImage * params.weakVisualImageMultiplier
    : params.wImage;

  const scores = new Map<string, number>();
  const add = (id: string, value: number) => {
    scores.set(id, (scores.get(id) ?? 0) + value);
  };

  if (params.mode === "weighted") {
    const normImage = minMaxNormalize(imageList);
    const normText = minMaxNormalize(textList);
    for (const [id, v] of normImage) add(id, effectiveImageWeight * v);
    for (const [id, v] of normText) add(id, params.wText * v);
  } else {
    // RRF
    const k = params.rrfK;
    for (const item of imageList) {
      add(item.product_id, (effectiveImageWeight * 1) / (k + item.rank));
    }
    for (const item of textList) {
      add(item.product_id, (params.wText * 1) / (k + item.rank));
    }
  }

  const fused: FusedItem[] = Array.from(scores.entries()).map(
    ([product_id, score]) => ({ product_id, score }),
  );

  // Stable sort by score desc, then product_id asc for deterministic ties.
  fused.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.product_id < b.product_id ? -1 : a.product_id > b.product_id ? 1 : 0;
  });
  return fused;
}

/** Min-max normalize a ranked list's scores to [0,1] (by product_id). */
function minMaxNormalize(list: RankedItem[]): Map<string, number> {
  const out = new Map<string, number>();
  if (list.length === 0) return out;
  let min = Infinity;
  let max = -Infinity;
  for (const item of list) {
    if (item.score < min) min = item.score;
    if (item.score > max) max = item.score;
  }
  const range = max - min;
  for (const item of list) {
    // When all scores are equal, normalize to 1.0 (all equally relevant).
    const value = range === 0 ? 1 : (item.score - min) / range;
    out.set(item.product_id, value);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Task 9 — near-identical-generic tie-breaker
// ---------------------------------------------------------------------------

/** Product signals (from D1) used to break ties against OCR text. */
export interface TieBreakSignals {
  product_id: string;
  score: number;
  name: string | null;
  strength: string | null;
  manufacturer: string | null;
  barcode: string | null;
}

const DEFAULT_TIE_BREAK_DELTA = 0.05;

/** Resolve the tie-break score delta from env (default 0.05). */
export function tieBreakDeltaFromEnv(env: Env): number {
  return num(env.TIE_BREAK_DELTA, DEFAULT_TIE_BREAK_DELTA);
}

/** Normalize free text for token matching (lowercase, collapse non-alnum). */
function normalizeText(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

/**
 * Score how strongly a product's metadata matches the OCR text. Combines
 * OCR-extracted name/strength + manufacturer + barcode signals. Higher = better.
 *
 * - name: fraction of name tokens (len>=3) present in the OCR text
 * - strength: +1.5 if the normalized strength token appears (e.g. "650mg")
 * - manufacturer: +1.0 if present
 * - barcode: +2.0 if the barcode digits appear in the OCR text (strongest,
 *   near-unique signal). Barcode IS derivable here when the pack's printed
 *   barcode number is captured among the OCR tokens.
 */
export function tieBreakMatchScore(
  signals: TieBreakSignals,
  ocrText: string,
): number {
  const hay = normalizeText(ocrText);
  if (!hay) return 0;
  const haySquished = hay.replace(/\s+/g, "");
  let score = 0;

  if (signals.name) {
    const tokens = normalizeText(signals.name)
      .split(" ")
      .filter((t) => t.length >= 3);
    if (tokens.length) {
      const hits = tokens.filter((t) => hay.includes(t)).length;
      score += hits / tokens.length;
    }
  }

  if (signals.strength) {
    const s = normalizeText(signals.strength).replace(/\s+/g, "");
    if (s && haySquished.includes(s)) score += 1.5;
  }

  if (signals.manufacturer) {
    const m = normalizeText(signals.manufacturer);
    if (m && m.length >= 3 && hay.includes(m)) score += 1.0;
  }

  if (signals.barcode) {
    const b = signals.barcode.replace(/[^0-9]/g, "");
    if (b.length >= 6 && haySquished.includes(b)) score += 2.0;
  }

  return score;
}

/**
 * Reorder products that are within `delta` of each other (a near-tie cluster)
 * using the OCR-derived tie-break match score. Products outside a cluster keep
 * their fusion order; within a cluster, higher OCR match wins, falling back to
 * the original fused score for stability.
 *
 * `items` must be pre-sorted by `score` descending.
 */
export function applyTieBreak(
  items: TieBreakSignals[],
  ocrText: string,
  delta: number = DEFAULT_TIE_BREAK_DELTA,
): TieBreakSignals[] {
  if (items.length < 2 || !ocrText.trim() || delta <= 0) return items.slice();

  const result: TieBreakSignals[] = [];
  let i = 0;
  while (i < items.length) {
    // Grow a cluster while each next item is within `delta` of the cluster head.
    const head = items[i].score;
    let j = i + 1;
    while (j < items.length && head - items[j].score <= delta) j++;
    const cluster = items.slice(i, j);
    if (cluster.length > 1) {
      const matches = new Map<string, number>(
        cluster.map((c) => [c.product_id, tieBreakMatchScore(c, ocrText)]),
      );
      cluster.sort((a, b) => {
        const ma = matches.get(a.product_id) ?? 0;
        const mb = matches.get(b.product_id) ?? 0;
        if (mb !== ma) return mb - ma;
        // Fallback: preserve original fused score order (stable).
        return b.score - a.score;
      });
    }
    result.push(...cluster);
    i = j;
  }
  return result;
}
