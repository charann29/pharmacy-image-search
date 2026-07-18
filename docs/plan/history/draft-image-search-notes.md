# Visual Image Search — Notes for Main Agent

## Scope confirmation
- Additive to the **existing** keyword + semantic text search. The existing search is a black-box service we call and merge from — the plan never redesigns or rebuilds it (adapter interface in `worker/src/text_search_adapter.ts`).
- Primary interaction optimized: photo → product (pure image-to-image ANN). Text/OCR is a fusion signal, not the core engine.
- All secrets referenced by env-var name only (`IMAGEKIT_PUBLIC_KEY`, `IMAGEKIT_PRIVATE_KEY`, `CF_ACCOUNT_ID`, `CF_D1_DATABASE_ID`, `CF_D1_API_TOKEN`, `ML_SERVICE_SHARED_SECRET`). No literal values in any file; assume pasted values are being rotated.

## Real gating task
- **DINOv3 license review + HuggingFace gated access (Task 0)** is a genuine blocker for commercial production use, not a formality. Meta AI custom license + acceptable-use policy; gated approval can take days. Start it day one. The pipeline is encoder-agnostic (config `ENCODER`, `EMBEDDING_DIM`), so **SigLIP 2 (Apache-2.0)** is a real drop-in fallback and lets all build work proceed before approval lands.

## Research confirmed (2026-07 current)
- DINOv3 loads via `AutoImageProcessor`/`AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")`, needs `transformers>=4.56.0`; gated.
- PaddleOCR **3.x** uses `PaddleOCR().predict(img)` (not `.ocr()`), returns structured dicts with text/confidence/bbox; PP-OCRv5 default.
- Qdrant Python client (1.10.x+) supports Universal Query API, `Prefetch`, `FusionQuery(fusion=RRF)`, and `ScalarQuantization(ScalarQuantizationConfig(type=INT8))`. Primary image query stays a single dense ANN; RRF kept available for future multi-vector hybrid.
- Cloudflare D1 REST: `POST /accounts/{account_id}/d1/database/{database_id}/raw` (or `/query`), Bearer API token with `Account:D1:Edit`. Note: Worker binding API is preferred for hot-path row queries — plan uses the D1 binding in the Worker and the REST API only for backfill/control-plane.
- ImageKit server upload: `POST https://api.imagekit.io/v2/upload`, `Authorization: Basic base64(private_key)`, server-side only.

## Open questions (non-blocking — sane defaults chosen, confirm if desired)
1. Worker→GPU-service transport: shared-secret header (default) vs Cloudflare Tunnel vs mTLS. Plan defaults to shared secret with Tunnel documented as hardening.
2. Grouping pool for multiple images/product: max vs mean pooling — plan makes it configurable, default max.
3. Fusion default weights / RRF k — plan exposes as tunable env; needs eval-driven tuning in Phase 3.

## Blocking questions
None that require pausing before the main agent submits — the stack is user-confirmed and the one true gate (DINOv3 license) is captured as Task 0 with a fully-specified fallback path. If the main agent wants to pre-empt, the single decision worth surfacing to the user is:
- **DINOv3 vs SigLIP2 as launch encoder** — (A) wait for DINOv3 gated approval for best image-to-image recall, accepting a few-days delay and custom-license terms; (B) launch on SigLIP2 (Apache-2.0) now and swap to DINOv3 later once approved. The encoder-agnostic design makes B→A cheap. Not strictly blocking because the plan supports both, but it affects launch timing.

## Recommended plan splits
Current scope is one coherent product feature (visual image search) and is appropriately sized as a single plan with 13 tasks; a split is **not** required. If the main agent prefers finer parallelization for dispatch, the only natural seam is:
- **Split A — ML tier** (Tasks 1–6, 9–11): FastAPI + encoders + Qdrant + backfill + OCR + eval + cost. Python/GPU ownership.
- **Split B — Edge tier** (Tasks 7, 8, 12): Cloudflare Worker + fusion + D1/ImageKit integration + rollout flags. TS/Cloudflare ownership.
- Split B depends on Split A's `/embed` `/search` `/ocr` contracts (defined in Task 3/6), so B integrates against those interfaces after they land. This is optional; the single plan already marks parallelism with `[parallel]`/`[after N]`.

## UI/frontend handoff
The plan's user-facing surface is a JSON API + Worker response shaping — no dedicated UI screens are in scope here. If a customer-facing "search by photo" UI (camera capture, results grid, confidence/fallback messaging) is wanted, that is a **separate UI/frontend plan** and needs a design subagent + Design-tab artifacts before user-visible submission. Flagging for the main orchestrator to decide whether to dispatch it.

## Self-review
- Every one of the 12 requested task areas maps to a numbered task (plus Task 0 for the license gate). ✔
- No placeholders/TODOs; every task has file paths + per-task verification. ✔
- Encoder/field/file-path names consistent across detailed plan. ✔
- Every task has `[parallel]`/`[after N]` marker + test expectation. ✔
- No submission/approval/reply/implementation actions included. ✔
- Secrets referenced by env-var name only; DINOv3 license is a real gating task. ✔
