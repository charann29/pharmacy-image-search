"""Live end-to-end integration probe (read-only against production).

Pulls a handful of REAL products + image URLs from the live Cloudflare D1
catalog (SELECT only), fetches the REAL images from ImageKit, embeds them with a
deterministic fake encoder (no GPU in this sandbox), indexes them into a
SCRATCH Qdrant collection, runs a nearest-neighbour query, asserts the query
image's own product ranks #1, then DELETES the scratch collection.

Nothing in production D1 or ImageKit is mutated. Secrets come from env only and
are never printed.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import time

import httpx
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

CF_ACCOUNT_ID = os.environ["CF_ACCOUNT_ID"]
CF_D1_DATABASE_ID = os.environ["CF_D1_DATABASE_ID"]
CF_D1_API_TOKEN = os.environ["CF_D1_API_TOKEN"]

DIM = 128
COLLECTION = "live_probe_scratch"
N_PRODUCTS = 20


def d1_query(sql: str, params=None) -> list[dict]:
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


def fake_embed(img: Image.Image) -> np.ndarray:
    """Deterministic content-based embedding: seed an RNG from image bytes.

    Same image -> same vector (cosine 1.0); different images -> different
    vectors. Stands in for DINOv3/SigLIP2 which can't run without a GPU.
    """
    buf = io.BytesIO()
    img.convert("RGB").resize((64, 64)).save(buf, format="PNG")
    seed = int.from_bytes(hashlib.sha256(buf.getvalue()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def main() -> int:
    print("== 1. Live D1: pull real products with images (SELECT only) ==")
    rows = d1_query(
        "SELECT pi.id AS image_id, pi.product_id, pi.url, p.name "
        "FROM product_images pi JOIN products p ON p.id = pi.product_id "
        "WHERE pi.url IS NOT NULL LIMIT ?",
        [N_PRODUCTS],
    )
    print(f"   fetched {len(rows)} image rows from live catalog")
    assert rows, "no image rows returned from live D1"
    print("== 2. Fetch REAL images from ImageKit + fake-embed ==")
    points = []
    fetched = []
    for row in rows:
        ir = httpx.get(row["url"], timeout=30.0, follow_redirects=True)
        if ir.status_code != 200:
            print(f"   skip image_id={row['image_id']} http={ir.status_code}")
            continue
        img = Image.open(io.BytesIO(ir.content))
        vec = fake_embed(img)
        points.append(
            PointStruct(
                id=int(row["image_id"]),
                vector=vec.tolist(),
                payload={"product_id": row["product_id"], "name": row["name"]},
            )
        )
        fetched.append((row, vec))
        print(f"   image_id={row['image_id']} product_id={row['product_id']} "
              f"bytes={len(ir.content)} name={str(row['name'])[:30]!r}")
    assert len(points) >= 3, "need at least 3 live images indexed"

    print("== 3. Index into SCRATCH Qdrant collection ==")
    qc = QdrantClient(url="http://localhost:6333")
    qc.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )
    qc.upsert(collection_name=COLLECTION, points=points)
    count = qc.count(collection_name=COLLECTION).count
    print(f"   indexed {count} vectors")
    assert count == len(points)

    print("== 4. Query with a real indexed image -> expect rank-1 self ==")
    # Pick a probe whose vector is unique in the set (some catalog entries share
    # a byte-identical placeholder image, which legitimately collide under any
    # content-based encoder). Choosing a unique image makes rank-1 self exact.
    keys = [tuple(np.round(v, 6)) for _, v in fetched]
    probe_idx = next(
        (i for i, k in enumerate(keys) if keys.count(k) == 1), 0
    )
    probe_row, probe_vec = fetched[probe_idx]
    t0 = time.time()
    hits = qc.query_points(
        collection_name=COLLECTION, query=probe_vec.tolist(), limit=3
    ).points
    dt = (time.time() - t0) * 1000
    top = hits[0]
    print(f"   query image_id={probe_row['image_id']} "
          f"product_id={probe_row['product_id']}")
    print(f"   top hit product_id={top.payload['product_id']} "
          f"score={top.score:.4f} latency={dt:.1f}ms")
    assert top.payload["product_id"] == probe_row["product_id"], "rank-1 mismatch"
    assert top.score > 0.99, f"self-similarity too low: {top.score}"
    print(f"   (note: {sum(1 for k in keys if keys.count(k) > 1)} of "
          f"{len(keys)} sampled images are byte-identical placeholders)")

    print("== 5. Cleanup scratch collection ==")
    qc.delete_collection(collection_name=COLLECTION)
    print("   deleted", COLLECTION)

    print("\nLIVE E2E: PASSED — real D1 read + real ImageKit fetch + "
          "Qdrant index/query, production untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
