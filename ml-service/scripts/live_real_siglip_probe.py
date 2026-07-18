"""REAL SigLIP 2 visual-search test against live catalog images (CPU).

Uses the actual production encoder class (app.encoders.siglip2.SigLIP2Encoder,
model google/siglip2-so400m-patch16-naflex) — no fake vectors. Pulls real
product image URLs from live D1 (read-only), fetches the real images from
ImageKit, embeds them with the real model on CPU, indexes into a SCRATCH Qdrant
collection, and runs visual-similarity queries. Production is never mutated;
the scratch collection is deleted at the end.

Secrets come from env only; nothing is printed.
"""
from __future__ import annotations

import io
import os
import sys
import time

import httpx
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CF_ACCOUNT_ID = os.environ["CF_ACCOUNT_ID"]
CF_D1_DATABASE_ID = os.environ["CF_D1_DATABASE_ID"]
CF_D1_API_TOKEN = os.environ["CF_D1_API_TOKEN"]

COLLECTION = "live_real_siglip_scratch"
N = int(os.environ.get("N_IMAGES", "40"))


def d1_query(sql, params=None):
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}"
        f"/d1/database/{CF_D1_DATABASE_ID}/query"
    )
    r = httpx.post(
        url,
        headers={"Authorization": f"Bearer {CF_D1_API_TOKEN}"},
        json={"sql": sql, "params": params or []},
        timeout=30.0,
    )
    r.raise_for_status()
    body = r.json()
    assert body.get("success"), body.get("errors")
    return body["result"][0]["results"]


def main() -> int:
    from app.encoders.siglip2 import SigLIP2Encoder
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    print("== 1. Load REAL SigLIP 2 model (CPU) ==")
    t0 = time.time()
    enc = SigLIP2Encoder()
    # force a load with a tiny dummy so timing of model load is explicit
    _ = enc.embed([Image.new("RGB", (64, 64), (128, 128, 128))])
    print(f"   model={enc.model_id} dim={enc.dim} loaded+warmup in {time.time()-t0:.1f}s")

    print(f"== 2. Pull {N} real image rows from live D1 (SELECT only) ==")
    rows = d1_query(
        "SELECT pi.id AS image_id, pi.product_id, pi.url, p.name "
        "FROM product_images pi JOIN products p ON p.id = pi.product_id "
        "WHERE pi.url IS NOT NULL LIMIT ?",
        [N],
    )
    print(f"   fetched {len(rows)} rows")

    print("== 3. Fetch real ImageKit images + REAL embeddings ==")
    imgs, meta = [], []
    for row in rows:
        ir = httpx.get(row["url"], timeout=30.0, follow_redirects=True)
        if ir.status_code != 200:
            continue
        try:
            img = Image.open(io.BytesIO(ir.content)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"   skip image_id={row['image_id']}: {e}")
            continue
        imgs.append(img)
        meta.append(row)
    t0 = time.time()
    vecs = enc.embed(imgs)  # (N, dim), L2-normalized by base class
    dt = time.time() - t0
    print(f"   embedded {len(imgs)} real images in {dt:.1f}s "
          f"({dt/max(len(imgs),1)*1000:.0f} ms/img on CPU), shape={vecs.shape}")
    assert vecs.shape[1] == enc.dim

    print("== 4. Index into SCRATCH Qdrant collection ==")
    qc = QdrantClient(url="http://localhost:6333")
    if qc.collection_exists(COLLECTION):
        qc.delete_collection(COLLECTION)
    qc.create_collection(
        COLLECTION, vectors_config=VectorParams(size=enc.dim, distance=Distance.COSINE)
    )
    qc.upsert(
        COLLECTION,
        points=[
            PointStruct(
                id=int(m["image_id"]),
                vector=vecs[i].tolist(),
                payload={"product_id": m["product_id"], "name": m["name"]},
            )
            for i, m in enumerate(meta)
        ],
    )
    print(f"   indexed {qc.count(COLLECTION).count} REAL vectors")

    print("== 5. Visual search: self-match sanity ==")
    # Some catalog entries share a byte-identical placeholder image (they embed
    # to the same vector, which is correct). Pick a probe whose image bytes are
    # unique so rank-1 self-match is unambiguous.
    import hashlib
    digs = [hashlib.sha256(i.tobytes()).hexdigest() for i in imgs]
    probe_i = next((k for k, d in enumerate(digs) if digs.count(d) == 1), 0)
    hits = qc.query_points(COLLECTION, query=vecs[probe_i].tolist(), limit=5).points
    print(f"   query: {str(meta[probe_i]['name'])[:35]!r} (pid {meta[probe_i]['product_id']})")
    for h in hits:
        print(f"     -> {h.score:.4f}  pid {h.payload['product_id']}  "
              f"{str(h.payload['name'])[:35]!r}")
    assert hits[0].payload["product_id"] == meta[probe_i]["product_id"]
    assert hits[0].score > 0.99, f"self score too low {hits[0].score}"
    dup_groups = sum(1 for d in set(digs) if digs.count(d) > 1)
    print(f"   self-match OK (rank-1, score ~1.0); "
          f"{dup_groups} placeholder-duplicate image group(s) in sample")

    print("== 6. Visual search: cross-image nearest neighbours (real similarity) ==")
    # For 3 probes, show top-3 *other* products the real model considers closest.
    for probe_i in [0, len(meta) // 2, len(meta) - 1]:
        hits = qc.query_points(
            COLLECTION, query=vecs[probe_i].tolist(), limit=4
        ).points
        others = [h for h in hits if h.id != int(meta[probe_i]["image_id"])][:3]
        print(f"   [{str(meta[probe_i]['name'])[:30]!r}] closest others:")
        for h in others:
            print(f"     {h.score:.4f}  {str(h.payload['name'])[:35]!r}")

    print("== 7. Cleanup ==")
    qc.delete_collection(COLLECTION)
    print("   deleted", COLLECTION)
    print("\nREAL SigLIP2 LIVE E2E: PASSED — actual model embeddings on real "
          "catalog images, production untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
