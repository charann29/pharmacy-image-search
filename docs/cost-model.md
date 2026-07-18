# Cost Model (Task 12)

Open-source/self-hosted vs. managed cost analysis for the visual image-search
tier. All figures are **engineering estimates** to size hardware and set
expectations; they must be **cross-checked against measured throughput** from the
backfill pipeline (Task 11) and current provider pricing. Secrets are referenced
by env-var **name** only.

**Fixed inputs:** ~80k products, **~320k images**. Encoder dim depends on the
active encoder:
- **DINOv3** (ViT-B/16): **dim ≈ 768**
- **SigLIP 2** (fallback / Phase-1 launch): **dim ≈ 1152**

Throughout, `N = 320,000` vectors.

---

## 1. Qdrant memory — broken out correctly

A common mistake is to size Qdrant as just `N × dim × 1 byte` (the int8 quantized
vectors). That **undercounts**. With `int8` scalar quantization **and**
`rescore=True`, Qdrant keeps the **original float32 vectors** too (used to
re-score the quantized candidates for accuracy), **plus** the HNSW graph,
**plus** payload + payload indexes, **plus** snapshot storage on disk. Break it
out:

### 1.1 Components

| Component | Formula | In RAM? |
|---|---|---|
| **Quantized int8 vectors** | `N × dim × 1 byte` | **Yes** (`always_ram=True`) |
| **Original float32 vectors** (for rescore) | `N × dim × 4 bytes` | RAM **or** on-disk (`on_disk=true`) |
| **HNSW graph** | `≈ N × M × 2 links × 4 bytes` (layer-0 dominates; `M=16`) | **Yes** |
| **Payload** `{product_id, image_id, encoder, is_reference}` | `≈ N × ~80–120 bytes` | RAM (or on-disk) |
| **Payload indexes** (e.g. `product_id`) | small — depends on cardinality (~80k product_ids) | RAM |
| **Snapshot storage** | `≈ float32 vectors + payload` (compressed on disk) | Disk only |

HNSW graph estimate: with `M=16`, layer-0 has up to `2×M = 32` neighbor links per
point at 4 bytes each → `N × 32 × 4 = N × 128 bytes`. For `N=320k` that is
`320,000 × 128 ≈ 41 MB` (upper-layer links add a small fraction). This is
**dim-independent**.

### 1.2 DINOv3 — dim = 768

- **Quantized int8:** `320,000 × 768 × 1 = 245,760,000 B ≈ 246 MB (~0.23 GiB)`
- **Float32 (rescore):** `320,000 × 768 × 4 = 983,040,000 B ≈ 983 MB (~0.92 GiB)`
- **HNSW graph:** `≈ 41 MB`
- **Payload (~100 B):** `320,000 × 100 = 32 MB`
- **Payload index:** low tens of MB

| Placement | RAM needed |
|---|---|
| float32 **in RAM** (fastest rescore) | `246 + 983 + 41 + 32 ≈ 1.30 GB` |
| float32 **on disk** (rescore reads disk) | `246 + 41 + 32 ≈ 0.32 GB` RAM + `~0.98 GB` disk |
| **Snapshot on disk** | `≈ 1.0 GB` |

### 1.3 SigLIP 2 — dim = 1152

- **Quantized int8:** `320,000 × 1152 × 1 = 368,640,000 B ≈ 369 MB (~0.34 GiB)`
- **Float32 (rescore):** `320,000 × 1152 × 4 = 1,474,560,000 B ≈ 1.47 GB (~1.37 GiB)`
- **HNSW graph:** `≈ 41 MB` (unchanged — depends on N and M, not dim)
- **Payload (~100 B):** `32 MB`

| Placement | RAM needed |
|---|---|
| float32 **in RAM** | `369 + 1,474 + 41 + 32 ≈ 1.92 GB` |
| float32 **on disk** | `369 + 41 + 32 ≈ 0.44 GB` RAM + `~1.47 GB` disk |
| **Snapshot on disk** | `≈ 1.5 GB` |

### 1.4 Sizing takeaway

- **Both encoders coexisting** (per-encoder collections during a swap): sum both.
  In-RAM float32 for both ≈ `1.30 + 1.92 ≈ 3.2 GB`.
- **RAM recommendation:** a Qdrant VM with **4 GB RAM** comfortably holds either
  single collection with float32 in RAM; **8 GB** covers both collections
  coexisting during an encoder swap, plus OS/Qdrant overhead and headroom. Put
  float32 **on disk** if you want to shrink RAM further at a small rescore-latency
  cost.
- **Disk:** budget a few GB per collection for the on-disk float32 (if used) +
  snapshots. Trivial.

> Memory is **not** the constraint at 320k vectors — even the largest breakdown is
> ~2 GB. The GPU (for embedding) and I/O are the real cost drivers.

---

## 2. One-time GPU indexing (full re-index of 320k images)

Indexing = fetch resized ImageKit transform → decode → GPU-embed in batches →
upsert to Qdrant + write `embedding_map`. The bottleneck is GPU embedding
throughput.

**Assumption (to be replaced by measured Task 11 throughput):** a modest
L4/T4-class GPU embedding a ViT-B/16-class model at fp16 with batching sustains,
end-to-end (including image fetch/decode overhead), somewhere in
**~50–200 images/sec**. Take a working point of **~100 img/s**.

```
GPU-hours = N / throughput / 3600
  @  50 img/s: 320,000 /  50 / 3600 = 1.78 h
  @ 100 img/s: 320,000 / 100 / 3600 = 0.89 h
  @ 200 img/s: 320,000 / 200 / 3600 = 0.44 h
```

So a **full re-index is well under a few GPU-hours**. Add margin for retries,
resume, and network I/O → **budget ~2–4 wall-clock hours** for the first backfill.

**GPU cost of one full index:** at an on-demand L4 rate of ~$0.5–0.8/hr, a full
re-index is roughly **$1–$3 of GPU time**. Re-embeds (packaging redesigns,
encoder swaps) cost the same per full pass, or a fraction when filtered to
changed products.

> **Emit and record actual throughput** at the end of `backfill_index.py`
> (GPU-hours + images/sec) and update this section with the measured number.

---

## 3. Storage & infra recurring costs

| Item | Estimate | Notes |
|---|---|---|
| **Catalog images on ImageKit** | **~160–320 GB** | 320k images × ~0.5–1 MB avg. Already stored by the user — reused, not new. Index-time fetches use **resized transforms** (e.g. 512px) to cut bandwidth. |
| **Qdrant VM** | small always-on instance | 4–8 GB RAM (see §1.4), a few GB disk. Can co-locate with the ML service on the GPU box, or a tiny separate VM. |
| **GPU host (steady state)** | on-demand/spot for bursty index; small always-on for query | Per-query = single embed + ANN (single-digit to tens of ms). See `docs/provider-alternatives.md` §4 for RunPod / AWS g6 / etc. |
| **Cloudflare Worker** | free/low tier | Requests + rate-limiting binding; edge orchestration is cheap. |
| **Cloudflare D1** | free/low tier | ~80k products + ~320k image rows + embedding_map is a small SQLite DB; reads/writes well within low-cost tiers. |

The dominant recurring cost is the **GPU host**, and its effective cost depends
entirely on **utilization** — a box that only serves light query traffic most of
the day is mostly idle.

---

## 4. Managed contrast (Vertex multimodal embeddings)

Managed embedding (Vertex `multimodalembedding`, ~**$0.0001/image**) replaces the
encoder only — you **still** run a vector DB.

- **One-time embed of the catalog:** `320,000 × $0.0001 = $32`. Cheap once.
- **Every re-embed** (packaging redesign batches, encoder/model version bump,
  re-index): another per-image charge on the affected images (a full re-embed =
  another $32).
- **Per-query fee (recurring, scales with traffic):** every user query image is
  embedded on Vertex. Example: `10,000 queries/day × $0.0001 = $1/day ≈ $30/mo`;
  `100,000 queries/day ≈ $10/day ≈ $300/mo`. This is the cost that never stops.
- **Plus** you still pay for the vector DB host (managed or self-hosted).

### 4.1 Open-source vs managed — explicit trade-off

| | Self-hosted (chosen) | Managed (Vertex) |
|---|---|---|
| Index 320k once | ~$1–3 GPU time | ~$32 |
| Re-embed (full) | ~$1–3 GPU time | ~$32 each |
| **Per-query cost** | **$0** (embed on our GPU) | **~$0.0001/query, forever** |
| Fixed monthly | GPU VM + tiny Qdrant VM | vector-DB host only |
| Lock-in | Low (open models, cloud-agnostic) | High (model-specific embeddings) |
| Ops burden | We own it (mitigated by Docker + runbook) | Near-zero |
| Data/PII | Photos stay in our environment | Photos sent to Google |

**Break-even is driven by query volume + GPU utilization, not the one-time
index.** Vertex's $32 one-time embed is cheap; its **per-query fee is the
recurring cost that self-hosting eliminates**. Self-hosting wins when query
volume is meaningful and the GPU is reasonably utilized (or shared with other
workloads); managed wins when traffic is very low/bursty and the fully-loaded
cost of owning a GPU + Qdrant exceeds the per-query fees you'd otherwise pay.

> **Validation required:** these self-hosted-vs-managed savings depend on
> **measured GPU VM utilization and actual query volume**. Re-run this comparison
> with the throughput emitted by `backfill_index.py` and observed production QPS
> before committing to a long-term GPU reservation. If traffic assumptions change,
> revisit — the managed path (Vertex + a managed vector DB) remains the documented
> escape hatch.
