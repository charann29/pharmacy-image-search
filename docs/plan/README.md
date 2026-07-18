# Implementation plan & history

- **`implementation-plan.md`** — the **approved** plan (v2), 14 tasks (Task 0–13).
  This is the plan the build followed.
- **`implementation-plan-summary.md`** — condensed summary of the approved plan.
- **`history/`** — full drafting trail for provenance:
  - `draft-image-search-detailed.md`, `draft-image-search-notes.md` — initial
    plan-subagent drafts.
  - `v1-image-search*.md` — first submitted version (superseded).
  - `v2-image-search*.md` — approved version (same content as
    `implementation-plan*.md`, kept here for the version record).

## Key decisions captured in the plan
- Self-hosted open-source stack (no per-image managed-API fees, no lock-in).
- Primary interaction: photo → product (pure image-to-image).
- Primary encoder **DINOv3** (SOTA image-to-image), **SigLIP 2** (Apache-2.0) as
  launch encoder + documented drop-in fallback; encoder-agnostic pipeline.
- Storage/CDN ImageKit; metadata Cloudflare D1; edge Cloudflare Worker; ML tier
  cloud-agnostic Docker (FastAPI + Qdrant) on a GPU host.
- Scope: image search only, additive to the existing keyword + semantic text
  search (treated as a black box behind an adapter).
