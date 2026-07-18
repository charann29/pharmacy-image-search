# Provider Alternatives (Task 12)

This document answers the provider/build-vs-buy questions directly: **can we use
Google (Cloud Vision / Vertex) instead?**, **which vector DB?**, **where do we
host the GPU?**, and **why the chosen stack** (ImageKit + Cloudflare D1 + Worker +
self-hosted GPU). All secrets are referred to by env-var **name** only.

**Context recap:** ~80k products, ~320k catalog images, image-to-image visual
similarity fused with the existing keyword+semantic text search. The image
encoder is self-hosted (DINOv3, SigLIP 2 fallback); OCR via PaddleOCR; vectors in
self-hosted Qdrant.

---

## 1. Google Cloud Vision — what it is and is not

Google Cloud **Vision API** provides OCR (text detection), label/landmark/logo
detection, object localization, safe-search, and web-entity matching.

**Key point:** Cloud Vision is **not** a product visual-similarity retrieval
engine over *your* catalog. It tells you *what is in* an image (labels, text,
logos); it does **not** return "which of my 80k products is this a photo of."
To do product retrieval you would **still build vector search** (embed your
catalog, index it, ANN-query it) — Cloud Vision does not remove that work.

- Google's closest managed retrieval offering was **Vision Product Search**
  (part of the Vision Product/Retail line), oriented at retail catalogs
  (apparel/home goods/general). It is a managed, per-call, catalog-indexed
  similarity service — but it is a **Google-hosted, per-request, lock-in**
  path, it is not tuned for pharmacy packaging, and it does not fuse with our
  existing text search. It replaces our vector tier entirely rather than
  complementing it.
- **Where Cloud Vision *could* fit:** as a **managed OCR alternative to
  PaddleOCR**. If we did not want to self-host OCR, Cloud Vision text detection
  (or Google Document AI) could produce the name/strength text we feed to the
  existing keyword search. Trade-off: per-image OCR fees + sending label photos
  (potential PII) to Google vs. free self-hosted PaddleOCR on the GPU we already
  run. Secret would be referenced as e.g. `GOOGLE_APPLICATION_CREDENTIALS` /
  `GCP_VISION_API_KEY` (name only).

**Verdict:** Cloud Vision does not replace our retrieval engine. It is at most an
optional managed OCR swap. It does not answer the core "which product is this."

---

## 2. Vertex AI multimodal embeddings — managed embedding alternative

Google **Vertex AI** offers a **multimodal embedding** model
(`multimodalembedding`) that produces image (and text) embeddings via a managed
API. This is the real managed alternative to our self-hosted encoder.

- **How it would slot in:** call Vertex to embed each catalog image (index time)
  and each query image (query time), store/search the vectors in a vector DB
  (still needed — Vertex embeds, it does not index/search). So Vertex replaces
  **only the encoder**, not Qdrant.
- **Pricing:** roughly **~$0.0001 per image** embedded (order-of-magnitude;
  confirm current Vertex pricing). See `docs/cost-model.md` for the ~$32
  one-time-embed math and recurring costs.

**Pros**
- **Zero ML maintenance:** no GPU host, no CUDA/driver/torch pinning, no model
  weights, no license gate (avoids the DINOv3 gating problem entirely).
- Strong, well-maintained embedding quality; managed autoscaling.

**Cons**
- **Per-image fee at index time** and **per-query fee at query time** — recurring
  cost that scales with traffic and every re-embed/re-index.
- **Vendor lock-in:** embeddings are model-specific; migrating off Vertex means a
  full re-embed of all 320k images.
- **Fixed embedding dim** dictated by Google; no control over the encoder.
- Sends every query photo (and catalog image) to Google — a **PII/data-residency**
  consideration for pharmacy label photos.
- Query latency includes a network round-trip to Google per query.

**Verdict:** viable managed fallback (documented in the runbook and cost model),
but recurring per-image + per-query fees and lock-in run counter to the user's
"existing accounts, no per-image fees, no lock-in" preference. We keep it as the
documented **managed escape hatch** if self-hosting ops burden proves too high.

---

## 3. Self-hosted vector DB options (at ~320k vectors)

At ~320k vectors of dim ≈768–1152, this is a **small-to-modest** index — it fits
comfortably in RAM on a single node with any of these. The choice is about ops
ergonomics, quantization/rescore support, and filtering, not raw scale.

| Option | What it is | Pros at 320k | Cons | Fit |
|---|---|---|---|---|
| **Qdrant** *(chosen)* | Rust vector DB, HNSW, service + REST/gRPC | First-class **int8 scalar quantization + rescore**, payload filtering, easy Docker deploy, per-collection config, snapshots | One more service to run | **Chosen** — quantization+rescore and payload filtering match our needs; trivial at this scale |
| **Milvus** | Distributed vector DB | Scales to billions; rich index types | **Heavyweight** (etcd/MinIO/pulsar deps); overkill and higher ops burden at 320k | Over-engineered here |
| **pgvector** | Postgres extension | **Simplest if already on Postgres**; transactional; one system | Weaker at scale/latency; HNSW+quantization less mature than Qdrant; we are **not** on Postgres (catalog is D1) | Not a fit — no existing Postgres |
| **FAISS** | Meta ANN **library** (not a service) | Fastest raw ANN; battle-tested | **No service/persistence/filtering/CRUD** — we'd build the server, upserts, payload filtering, snapshots ourselves | Rejected — reinvents Qdrant |

**Why Qdrant wins here:** at 320k vectors the scale argument for Milvus is moot,
we have no Postgres to make pgvector "free," and FAISS would force us to build the
service layer Qdrant already gives us. Qdrant's int8-quant + rescore and payload
filtering (`product_id`, `is_reference`, `encoder`) are exactly what the pipeline
needs, and per-encoder collections make the DINOv3↔SigLIP2 swap clean.

---

## 4. GPU hosting options

We need one modest GPU (L4 / T4 / g6-class) for the encoder + OCR. Full re-index
of 320k images is a few GPU-hours (see `docs/cost-model.md`); steady-state
per-query is a single embed + ANN. So the workload is **bursty index + light
query** — utilization drives the cost decision.

| Option | Model | Pros | Cons |
|---|---|---|---|
| **RunPod** | On-demand / spot GPU pods, community & secure cloud | Cheap per-hour L4/A10/T4; good for **bursty indexing**; per-second billing | Less "enterprise" SLA; storage/networking ergonomics vary; need to manage the pod |
| **Lambda (Lambda Labs)** | On-demand cloud GPUs | Simple, GPU-focused, competitive pricing | Capacity availability can be tight; fewer managed extras |
| **AWS g6 / g6e** | EC2 with NVIDIA L4 | Mature ecosystem, VPC/IAM, spot for indexing, near existing infra | Higher on-demand $/hr; ops overhead; egress fees |
| **GCP (g2 / L4)** | Compute Engine + L4 | Integrates if we later use Vertex; committed-use discounts | Similar cost/ops to AWS; lock-in gravity toward GCP |
| **Azure (NC/NV series)** | NVIDIA GPUs on Azure | Fine if org is Azure-centric | Not our stack; comparable cost/ops |
| **Self-managed VM / on-prem** | Own hardware or bare-metal rental | Lowest steady-state cost at high utilization; full data control (pharmacy PII) | Cap-ex / provisioning; we own uptime, drivers, security patching |

**Recommendation:** cloud-agnostic Docker Compose (`infra/docker-compose.yml`)
means we are **not tied to any one provider**. Use a cheap on-demand/spot GPU
(RunPod or AWS g6 spot) for the **bursty backfill/re-index** jobs, and a small
always-on instance (or the same box) for steady-state query serving. Because the
stack is containerized with `gpus: all`, moving providers is a redeploy, not a
rewrite. Validate the actual $/month against **measured throughput and query
volume** (see cost model).

---

## 5. Why this stack (ImageKit + D1 + Worker + self-hosted GPU)

Given the user's existing accounts and stated priorities (**no per-image fees, no
vendor lock-in**), the chosen stack is:

- **ImageKit (storage + CDN + transforms):** the user **already** stores catalog
  images here. We reuse it for storage/CDN and request resized transforms (e.g.
  512px) at index time to cut embedding bandwidth. No new storage vendor; no
  per-image ML fee. (Secrets: `IMAGEKIT_PUBLIC_KEY`, `IMAGEKIT_PRIVATE_KEY`.)
- **Cloudflare D1 (catalog metadata):** serverless SQLite at the edge for
  `products` / `product_images` / `embedding_map`. Accessed via a Worker
  **binding** (no API token in the Worker); the indexing environment uses D1 REST
  with admin-scoped creds referenced by name (`CLOUDFLARE_API_TOKEN`,
  `CLOUDFLARE_ACCOUNT_ID`, `D1_DATABASE_ID`).
- **Cloudflare Worker (edge orchestration):** handles upload, D1 hydration,
  authenticated calls to the GPU service, hybrid fusion, rate limiting, and
  response shaping — close to the user, cheap, and already in the user's
  ecosystem. (Secrets: `ML_SERVICE_URL`, `ML_SERVICE_SHARED_SECRET`,
  `TEXT_SEARCH_URL`, `TEXT_SEARCH_AUTH`, `COMMERCE_API_URL`, `COMMERCE_API_AUTH`.)
- **Self-hosted GPU (encoder + OCR + Qdrant):** **no per-image / per-query fees**,
  **no lock-in** (open models + open Qdrant, cloud-agnostic Docker), and full
  control of pharmacy label-photo data (PII stays in our environment; query
  photos streamed, not persisted by default).

**Trade-off we are accepting:** self-hosting shifts uptime, scaling, and memory
sizing onto us. Mitigations: cloud-agnostic Docker, healthchecks, quantization,
the runbook, and a documented **managed fallback** (Vertex embeddings + a managed
vector DB) if the ops burden ever outweighs the savings. See `docs/cost-model.md`
for the break-even analysis — self-hosted savings depend on GPU VM utilization and
query volume, validated by measured throughput.

---

## 6. Summary table

| Concern | Managed (Google) | Chosen (self-hosted) |
|---|---|---|
| Product retrieval | Vision ≠ retrieval; Vertex embeds only (still need vector DB) | Self-hosted encoder + Qdrant |
| OCR | Cloud Vision / Document AI (per-image fee) | PaddleOCR (free, on our GPU) |
| Recurring cost | Per-image + per-query fees | GPU VM only; no per-item fees |
| Lock-in | High (model + platform) | Low (open models + Qdrant, cloud-agnostic) |
| Ops burden | Near-zero | We own it (mitigated by Docker + runbook) |
| Data/PII | Photos leave our environment | Photos stay in our environment |
