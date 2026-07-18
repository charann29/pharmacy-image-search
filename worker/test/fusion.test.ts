import { describe, it, expect } from "vitest";
import {
  fuse,
  fusionParamsFromEnv,
  rankMlHits,
  applyTieBreak,
  tieBreakMatchScore,
  tieBreakDeltaFromEnv,
  DEFAULT_FUSION_PARAMS,
  type TieBreakSignals,
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

describe("tieBreakMatchScore", () => {
  const sig = (o: Partial<TieBreakSignals>): TieBreakSignals => ({
    product_id: "P",
    score: 0,
    name: null,
    strength: null,
    manufacturer: null,
    barcode: null,
    ...o,
  });

  it("scores name token overlap with OCR text", () => {
    const s = tieBreakMatchScore(sig({ name: "Crocin Advance" }), "crocin box front");
    expect(s).toBeGreaterThan(0);
  });

  it("rewards strength match strongly", () => {
    const withStrength = tieBreakMatchScore(sig({ strength: "650mg" }), "tablet 650 mg pack");
    expect(withStrength).toBeGreaterThanOrEqual(1.5);
  });

  it("rewards barcode match most strongly", () => {
    const s = tieBreakMatchScore(sig({ barcode: "8901234567" }), "code 8901234567 back");
    expect(s).toBeGreaterThanOrEqual(2.0);
  });

  it("returns 0 for empty OCR text", () => {
    expect(tieBreakMatchScore(sig({ name: "Crocin" }), "  ")).toBe(0);
  });
});

describe("applyTieBreak", () => {
  const item = (id: string, score: number, o: Partial<TieBreakSignals> = {}): TieBreakSignals => ({
    product_id: id,
    score,
    name: null,
    strength: null,
    manufacturer: null,
    barcode: null,
    ...o,
  });

  it("reorders near-tie cluster by OCR match (barcode wins)", () => {
    // A and B are within delta; B's barcode is in the OCR text -> B first.
    const items = [
      item("A", 0.90, { name: "Generic Paracetamol" }),
      item("B", 0.88, { name: "Generic Paracetamol", barcode: "8901234567" }),
      item("C", 0.20, { name: "Unrelated" }),
    ];
    const out = applyTieBreak(items, "paracetamol 8901234567", 0.05);
    expect(out.map((i) => i.product_id)).toEqual(["B", "A", "C"]);
  });

  it("does not reorder products outside the delta cluster", () => {
    const items = [
      item("A", 0.90, { name: "Alpha" }),
      item("B", 0.50, { name: "Beta", barcode: "111222333" }), // strong match but far below
    ];
    // B matches OCR strongly but is outside delta of A -> order preserved.
    const out = applyTieBreak(items, "beta 111222333", 0.05);
    expect(out.map((i) => i.product_id)).toEqual(["A", "B"]);
  });

  it("no-ops without OCR text or with single item", () => {
    const items = [item("A", 0.9), item("B", 0.88)];
    expect(applyTieBreak(items, "", 0.05).map((i) => i.product_id)).toEqual(["A", "B"]);
    expect(applyTieBreak([item("A", 0.9)], "x", 0.05).map((i) => i.product_id)).toEqual(["A"]);
  });

  it("falls back to fused score order on equal OCR match", () => {
    const items = [item("A", 0.90), item("B", 0.89)];
    const out = applyTieBreak(items, "nomatch text", 0.05);
    expect(out.map((i) => i.product_id)).toEqual(["A", "B"]);
  });
});

describe("tieBreakDeltaFromEnv", () => {
  it("defaults to 0.05", () => {
    expect(tieBreakDeltaFromEnv({} as any)).toBe(0.05);
  });
  it("reads TIE_BREAK_DELTA", () => {
    expect(tieBreakDeltaFromEnv({ TIE_BREAK_DELTA: "0.1" } as any)).toBe(0.1);
  });
});
