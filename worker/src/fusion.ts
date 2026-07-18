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
  if (value === undefined || value === null || value === "") return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
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
