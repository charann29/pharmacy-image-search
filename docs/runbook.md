# Operations Runbook (Task 13)

Operational procedures for the visual image-search tier: phased rollout,
encoder swap, re-index, reconcile, and rollback. This is the ops source of truth.
Eval/index specifics (exact recall@k targets, calibrated threshold values) are
finalized by Task 10 and appended as they are measured — the structure and
concrete procedures below are complete now.

All secrets are referenced by **env-var name only** — never write literal values
into this or any file.

---

## 0. Feature flags & key config (the control surface)

| Name | Purpose | Rollback-relevant |
|---|---|---|
| `IMAGE_SEARCH_ENABLED` | Master on/off for image search. Off → pure text search. | **Yes — primary kill switch** |
| `HYBRID_FUSION_ENABLED` | Enable image+text fusion. Off → image-only (or text-only if image also off). | Yes |
| `ACTIVE_COLLECTION` | Which Qdrant collection the query path uses (e.g. `image_embeddings_siglip2_1152`). | Yes — encoder swap/rollback |
| `ENCODER` | `dinov3` \| `siglip2`. Selects the loaded model + its dim. | Yes |
| `EMBEDDING_DIM` | Encoder embedding dim (768 DINOv3 / 1152 SigLIP2). Must match the active collection. | Startup-validated |
| `W_IMAGE`, `W_TEXT`, `RRF_K` | Fusion tuning. | — |
| `weak_visual_match` threshold | Per-encoder confidence cutoff (calibrated by Task 10). | — |

**Startup consistency guard:** the ML service asserts on boot that
`ACTIVE_COLLECTION` exists, its vector dim == the loaded encoder's
`EMBEDDING_DIM`, and its recorded `encoder` metadata == `ENCODER`. A mismatch
**blocks startup** with a clear error rather than serving wrong queries. Keep
these three in sync on every change.

Secrets used by this tier (names only): `ML_SERVICE_URL`,
`ML_SERVICE_SHARED_SECRET`, `IMAGEKIT_PUBLIC_KEY`, `IMAGEKIT_PRIVATE_KEY`,
`TEXT_SEARCH_URL`, `TEXT_SEARCH_AUTH`, `COMMERCE_API_URL`, `COMMERCE_API_AUTH`,
`HUGGINGFACE_HUB_TOKEN` (indexing env, DINOv3 pull), `CLOUDFLARE_API_TOKEN` /
`CLOUDFLARE_ACCOUNT_ID` / `D1_DATABASE_ID` (indexing env D1 REST only — **not** in
the Worker; the Worker uses the D1 binding).

---

## 1. Phased rollout

### Phase 1 — MVP (image-only, SigLIP 2)
Goal: ship visual search without waiting on the DINOv3 license gate.
- **Encoder:** `ENCODER=siglip2` (Apache-2.0 — unblocked; see
  `docs/license-review-dinov3.md`). `EMBEDDING_DIM=1152`,
  `ACTIVE_COLLECTION=image_embeddings_siglip2_1152`.
- **Flags:** `IMAGE_SEARCH_ENABLED=true`, `HYBRID_FUSION_ENABLED=false`.
- Behavior: photo → embed → ANN → group → threshold → products. **No fusion** —
  image results returned directly (text search unchanged/parallel).
- Pre-reqs: catalog synced (Task 3), backfill indexed into the SigLIP2 collection
  (Task 11), thresholds at safe provisional defaults (Task 6) pending
  calibration.
- Ship gate: eval harness (Task 10) shows acceptable recall@k on the dev
  fixture; `/healthz` green; rate limiter active.

### Phase 2 — Hybrid fusion
Goal: fuse image + existing text search for better ranking.
- **Flags:** `IMAGE_SEARCH_ENABLED=true`, `HYBRID_FUSION_ENABLED=true`.
- Behavior: photo read once → fan out to `/search` (image) + `/ocr` in parallel →
  OCR text to the existing text-search adapter → **RRF** (default) / weighted
  fusion → D1 hydrate → price via commerce adapter → shaped response.
- On `weak_visual_match`, fusion down-weights image and leans on text.
- Graceful degradation: if the text adapter or commerce adapter times out, return
  image-only / omit price rather than failing the request.
- Ship gate: A/B (fused image+text vs text-only) on the eval set shows a positive
  delta; degradation paths verified.

### Phase 3 — Hardening
- **DINOv3 swap** once Task 0 clears (license signed off **and** HF access
  approved). Procedure: index the DINOv3 collection → **recalibrate per-encoder
  thresholds via Task 10** → flip `ACTIVE_COLLECTION` → run eval for
  parity/gain **before** production traffic. (Detailed steps in §2.)
- **Quantization tuning:** validate int8 + rescore recall vs latency; adjust
  `rescore` / HNSW `ef` if needed.
- **OCR/barcode tie-breakers:** enable fusion tie-breakers combining OCR
  name/strength + `manufacturer` + `barcode` when top visual scores are within a
  small delta (near-identical generics).
- **Re-embed cadence:** schedule periodic re-embed for packaging redesigns and
  new reference images (see §4).

---

## 2. Encoder-swap procedure (e.g. SigLIP 2 → DINOv3)

Per-encoder collections make this a **zero-downtime** swap: the new collection is
built alongside the live one, then the query path is flipped.

**Pre-req (DINOv3 only):** Task 0 complete — license signed off + HF access
approved; `HUGGINGFACE_HUB_TOKEN` provisioned in the indexing env. **Do not**
deploy DINOv3 weights to prod before this.

1. **Provision the new collection config:** set indexing-env `ENCODER=dinov3`,
   `EMBEDDING_DIM=768`; target collection `image_embeddings_dinov3_768`
   (`ensure_collection()` creates it: cosine, HNSW `m=16 ef_construct=128`, int8
   scalar quant `always_ram=True`).
2. **Backfill into the new collection:** run `backfill_index.py` for the DINOv3
   encoder. It is idempotent + resumable, writes deterministic vector IDs
   (`uuid5(ns, "dinov3:768:{image_id}")`) and `embedding_map` rows with
   `encoder=dinov3`. The **live SigLIP2 collection is untouched** and keeps
   serving.
3. **Reconcile** the new collection (see §4): Qdrant count == mapped count for
   `encoder=dinov3`; nearest neighbor of a known image is itself.
4. **Recalibrate thresholds:** run Task 10 eval against the DINOv3 collection to
   derive DINOv3-specific `weak_visual_match` thresholds and confirm recall@k
   parity/gain vs SigLIP2. Record the DINOv3 thresholds in config. **Do this
   before flipping** — thresholds are encoder-specific.
5. **Flip the query path:** set `ACTIVE_COLLECTION=image_embeddings_dinov3_768`
   and `ENCODER=dinov3`, `EMBEDDING_DIM=768` on the serving service. Restart; the
   startup guard verifies collection⇄encoder⇄dim consistency.
6. **Verify in prod-shadow / canary:** smoke-test `/search` on known photos;
   confirm eval parity; watch weak-match rate and latency.
7. **Keep the old collection** for at least one cycle as a warm rollback target
   (see §5); delete only after DINOv3 is proven.

---

## 3. Re-index procedure (full or partial)

Use when weights/model version change, images are added/redesigned, or a
collection must be rebuilt.

- **Full re-index:** run `backfill_index.py` for the target `ENCODER`. Idempotent
  (skips image_ids already mapped for that encoder) and resumable (cursor/
  checkpoint, retry w/ backoff). A killed run resumes to full count with no
  duplicate IDs. Emits throughput + GPU-hours at the end — **record these** and
  update `docs/cost-model.md`.
- **Partial / changed-products re-embed:** run `backfill_index.py` filtered to the
  changed `product_id`s (packaging redesigns, new reference images). Multiple
  reference images per product (`is_reference`) coexist; both old and new packs
  can match.
- **Bandwidth:** the pipeline fetches a **resized ImageKit transform** (e.g.
  512px), not the full-res original, to cut fetch/decode cost.
- Verify: Qdrant count == `embedding_map` count for the encoder; spot-check
  nearest-neighbor self-match.

---

## 4. Reconcile procedure (Qdrant ⇄ D1 consistency)

`reconcile.py` repairs partial-failure drift between Qdrant and D1.

- **Orphan vectors** (in Qdrant, no live D1 mapping) → delete from Qdrant.
- **Missing vectors** (mapped/active in D1, absent in Qdrant) → re-embed + upsert.
- **Deactivated products** (`active=0`, `product_images.deleted_at` set by
  `catalog_sync.py` soft-delete) → delete their vectors from Qdrant, **then** the
  soft-deleted D1 image rows may be hard-deleted (order matters: vectors first, so
  the mapping still exists when reconcile runs; FK `embedding_map.image_id` is
  `ON DELETE RESTRICT`).
- Run reconcile after every backfill and on a schedule. Success: Qdrant point
  count == mapped count for the encoder; no orphans; no missing.

---

## 5. Rollback

Rollback is **fast and flag-driven** — the whole tier is additive, so disabling it
returns the pre-existing pure text search.

**Escalating levers (least → most drastic):**

1. **Disable fusion:** `HYBRID_FUSION_ENABLED=false`. Falls back to image-only
   (or, combined with the next lever, text-only). Use when fusion ranking
   regresses but image search is fine.
2. **Disable image search entirely (primary kill switch):**
   `IMAGE_SEARCH_ENABLED=false`. The Worker returns **pure text-search behavior** —
   proves the feature is additive/non-breaking. Use for any image-tier incident
   (GPU down, bad index, latency spike).
3. **Roll back an encoder swap:** point `ACTIVE_COLLECTION` (and `ENCODER` /
   `EMBEDDING_DIM`) back to the previous collection
   (e.g. `image_embeddings_siglip2_1152`). This is why the old collection is
   **retained** through at least one cycle after a swap. Restart; startup guard
   re-validates consistency.
4. **Bad index / corruption:** flip `ACTIVE_COLLECTION` to a known-good collection
   or restore Qdrant from the latest **snapshot**, then re-run reconcile.

**Verification after any rollback:** toggling `IMAGE_SEARCH_ENABLED=false` yields
identical results to the legacy text-only path (regression check); `/healthz`
green; error/latency back to baseline.

---

## 6. Routine health checks

- `GET /healthz` on the ML service → 200 (encoder loaded, Qdrant reachable,
  active collection consistent).
- GPU availability on the host:
  `docker compose run --rm ml-service python -c "import torch; assert torch.cuda.is_available()"`.
- Qdrant collection point count vs `embedding_map` count per encoder (drift → run
  reconcile).
- Metrics (Task 10): per-stage latency (embed / ANN / fusion / e2e), weak-match
  rate, index freshness. Alert on weak-match-rate spikes (possible index/encoder
  mismatch) and latency-percentile regressions.

---

## 7. Appendix — filled by Task 10 / Task 11 as measured

- [ ] Calibrated `weak_visual_match` thresholds per encoder (SigLIP2, DINOv3).
- [ ] Recall@1/5/10 targets and measured values per encoder.
- [ ] Measured indexing throughput (img/s) + full-index GPU-hours (feeds
      `docs/cost-model.md`).
- [ ] Per-stage latency percentiles at production batch sizes.
- [ ] A/B deltas: fused image+text vs text-only.
