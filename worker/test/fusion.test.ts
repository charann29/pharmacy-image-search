import { describe, it, expect } from "vitest";
import {
  fuse,
  fusionParamsFromEnv,
  rankMlHits,
  DEFAULT_FUSION_PARAMS,
} from "../src/fusion";
import type { RankedItem } from "../src/types";

function ranked(ids: string[], baseScore = 1): RankedItem[] {
  return ids.map((id, i) => ({
    product_id: id,
    score: baseScore - i * 0.1,
    rank: i + 1,
  }));
}

describe("fusion — RRF", () => {
  it("fuses two known ranked lists into the expected order", () => {
    // Image: A(1), B(2), C(3); Text: B(1), C(2), D(3)
    const image = ranked(["A", "B", "C"]);
    const text = ranked(["B", "C", "D"]);
    const fused = fuse(image, text, false, {
      ...DEFAULT_FUSION_PARAMS,
      rrfK: 60,
    });
    // B appears rank2 image + rank1 text -> highest combined.
    // Compute expected: A=1/61; B=1/62+1/61; C=1/63+1/62; D=1/63
    const byId = Object.fromEntries(fused.map((f) => [f.product_id, f.score]));
    expect(byId["B"]).toBeGreaterThan(byId["C"]);
    expect(byId["C"]).toBeGreaterThan(byId["A"]);
    expect(byId["A"]).toBeGreaterThan(byId["D"]);
    expect(fused.map((f) => f.product_id)).toEqual(["B", "C", "A", "D"]);
  });

  it("is deterministic on ties (product_id asc)", () => {
    const image = ranked(["X", "Y"]);
    const text = ranked(["Y", "X"]); // symmetric -> tie
    const fused = fuse(image, text, false);
    expect(fused.map((f) => f.product_id)).toEqual(["X", "Y"]);
  });
});

describe("fusion — weighted weights", () => {
  it("text weight 0 -> image-only order", () => {
    const image = ranked(["A", "B", "C"]);
    const text = ranked(["C", "B", "A"]);
    const fused = fuse(image, text, false, {
      mode: "weighted",
      wImage: 1,
      wText: 0,
      rrfK: 60,
      weakVisualImageMultiplier: 0.3,
    });
    expect(fused.map((f) => f.product_id)).toEqual(["A", "B", "C"]);
  });

  it("image weight 0 -> text-only order", () => {
    const image = ranked(["A", "B", "C"]);
    const text = ranked(["C", "B", "A"]);
    const fused = fuse(image, text, false, {
      mode: "weighted",
      wImage: 0,
      wText: 1,
      rrfK: 60,
      weakVisualImageMultiplier: 0.3,
    });
    expect(fused.map((f) => f.product_id)).toEqual(["C", "B", "A"]);
  });
});

describe("fusion — weak_visual_match", () => {
  it("down-weights image and prefers text when weak", () => {
    // Image ranks A first; text ranks Z first (Z/W not in image list).
    const image = ranked(["A", "B"]);
    const text = ranked(["Z", "W"]);
    const strong = fuse(image, text, false, { ...DEFAULT_FUSION_PARAMS });
    const weak = fuse(image, text, true, {
      ...DEFAULT_FUSION_PARAMS,
      weakVisualImageMultiplier: 0.05,
    });
    // With strong visual, image rank-1 (A) ties text rank-1 (Z); A wins on
    // the deterministic product_id tie-break.
    expect(strong[0].product_id).toBe("A");
    // With weak visual match heavily down-weighting image, text's Z leads.
    expect(weak[0].product_id).toBe("Z");
  });
});

describe("fusionParamsFromEnv", () => {
  it("defaults to RRF with default k", () => {
    const p = fusionParamsFromEnv({} as any);
    expect(p.mode).toBe("rrf");
    expect(p.rrfK).toBe(DEFAULT_FUSION_PARAMS.rrfK);
  });
  it("reads weighted mode + tunables", () => {
    const p = fusionParamsFromEnv({
      FUSION_MODE: "weighted",
      W_IMAGE: "2",
      W_TEXT: "0.5",
      RRF_K: "30",
      WEAK_VISUAL_IMAGE_MULTIPLIER: "0.1",
    } as any);
    expect(p).toEqual({
      mode: "weighted",
      wImage: 2,
      wText: 0.5,
      rrfK: 30,
      weakVisualImageMultiplier: 0.1,
    });
  });
});

describe("rankMlHits", () => {
  it("assigns 1-based ranks in order", () => {
    const r = rankMlHits([
      { product_id: "A", score: 0.9 },
      { product_id: "B", score: 0.8 },
    ]);
    expect(r).toEqual([
      { product_id: "A", score: 0.9, rank: 1 },
      { product_id: "B", score: 0.8, rank: 2 },
    ]);
  });
});
