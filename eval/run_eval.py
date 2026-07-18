"""Image-search eval harness + threshold calibration + A/B report (Task 10).

Runs each labeled query photo through the real search path —
``get_encoder -> qdrant query -> grouping -> threshold`` — against an ephemeral
Qdrant (in-memory by default) seeded from the dataset's catalog manifest, then:

  * computes **recall@1/5/10**, **top-1 accuracy**, per-stage **latency**
    percentiles (embed / ANN / fusion / end-to-end), **weak-match rate**, and
    **index freshness** (max ``indexed_at`` vs catalog ``source_updated_at``);
  * **calibrates** the per-encoder ``weak_visual_match`` threshold against
    explicit targets (recall@1 >= target, false-positive rate <= limit) and
    writes the chosen thresholds to a JSON file the runbook references;
  * runs an **A/B** report comparing fused image+text vs text-only using an
    **injectable** text-results provider (the real text search is external).

Pluggable encoder: because torch/models/GPU are unavailable here, a
deterministic ``--fake-encoder`` mode downscales each image to a fixed vector so
the whole harness (metrics + calibration + A/B) is testable on the committed
fixture with no model downloads. Drop ``--fake-encoder`` to use the real
``get_encoder`` factory once models/GPU are present.

Usage::

    python eval/run_eval.py --dataset eval/dataset --fake-encoder
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

# Make ml-service importable (eval/ lives next to ml-service/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ML_SERVICE = os.path.join(os.path.dirname(_HERE), "ml-service")
if _ML_SERVICE not in sys.path:
    sys.path.insert(0, _ML_SERVICE)

from app.config import Settings  # noqa: E402
from app.search import grouping, qdrant_client as qc, threshold as thr  # noqa: E402

# eval/ is importable as a package-less module; import the sibling builder.
sys.path.insert(0, _HERE)
import build_eval_set as bes  # noqa: E402

# A text-results provider maps a QueryEntry to ranked text hits
# ``[{product_id, score, rank}]`` — injectable because the real text search is
# an external black box (Task 7 adapter).
TextProvider = Callable[["bes.QueryEntry"], list[dict[str, Any]]]

DEFAULT_TOP_K = 50
DEFAULT_RECALL_TARGET = 0.8
DEFAULT_FP_LIMIT = 0.2
DEFAULT_RRF_K = 60
# Multiplier applied to the image weight on weak_visual_match. MUST match the
# production Worker (worker/src/fusion.ts -> weakVisualImageMultiplier = 0.3) so
# eval-side fusion ranking (and thus threshold calibration) mirrors production.
DEFAULT_WEAK_VISUAL_MULTIPLIER = 0.3


# --------------------------------------------------------------------------
# Deterministic fake encoder (no model download / GPU)
# --------------------------------------------------------------------------
class FakeEncoder:
    """Deterministic encoder: downscale image -> flattened RGB -> L2-normalized.

    Produces the same vector for identical pixels and near vectors for the
    jittered query/catalog images of one product, so the harness exercises the
    real ANN + grouping + threshold path with meaningful, reproducible scores.
    """

    def __init__(self, grid: int = 8, encoder_name: str = "fake"):
        self._grid = grid
        self._name = encoder_name

    @property
    def dim(self) -> int:
        return self._grid * self._grid * 3

    @property
    def name(self) -> str:
        return self._name

    def embed(self, images: Sequence[Any]) -> np.ndarray:
        from app.encoders.base import l2_normalize
        from PIL import Image

        rows = []
        for img in images:
            small = img.convert("RGB").resize((self._grid, self._grid), Image.BILINEAR)
            rows.append(np.asarray(small, dtype=np.float32).reshape(-1) / 255.0)
        return l2_normalize(np.vstack(rows))


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------
@dataclass
class QueryResult:
    query_image_path: str
    expected_product_id: Optional[str]
    in_catalog: bool
    ranked_products: list[str]          # image-only ranking (product ids)
    top_score: float
    weak_visual_match: bool
    fused_ranked_products: list[str]    # image+text fused ranking
    text_ranked_products: list[str]     # text-only ranking


@dataclass
class LatencyStage:
    samples: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    def percentiles(self) -> dict[str, float]:
        if not self.samples:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0}
        arr = np.asarray(self.samples, dtype=np.float64)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(arr.mean()),
        }


# --------------------------------------------------------------------------
# Encoder selection
# --------------------------------------------------------------------------
def build_encoder(fake: bool, settings: Settings):
    """Return (encoder, encoder_name, dim). Fake mode avoids all model deps."""
    if fake:
        enc = FakeEncoder()
        return enc, enc.name, enc.dim
    from app.encoders import get_encoder

    enc = get_encoder(settings)
    return enc, settings.encoder, enc.dim


# --------------------------------------------------------------------------
# Text-provider (injectable; default = deterministic mock)
# --------------------------------------------------------------------------
def default_mock_text_provider(
    all_product_ids: Sequence[str],
    hit_rate: float = 0.7,
    seed: int = 7,
) -> TextProvider:
    """Build a deterministic mock text search.

    For an in-catalog query it usually (``hit_rate``) ranks the expected product
    first, otherwise returns a plausible-but-wrong ordering. Out-of-catalog
    queries get an arbitrary low-signal list. Deterministic per query path so
    A/B numbers are reproducible.
    """
    import hashlib

    products = list(all_product_ids)

    def _provider(q: "bes.QueryEntry") -> list[dict[str, Any]]:
        h = int(hashlib.sha256(f"{seed}:{q.query_image_path}".encode()).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        order = list(products)
        rng.shuffle(order)
        if q.expected_product_id and (h % 100) / 100.0 < hit_rate:
            # Promote the expected product to the front.
            order = [q.expected_product_id] + [p for p in order if p != q.expected_product_id]
        return [
            {"product_id": pid, "score": 1.0 - i / max(len(order), 1), "rank": i}
            for i, pid in enumerate(order[:DEFAULT_TOP_K])
        ]

    return _provider


# --------------------------------------------------------------------------
# Fusion (RRF) — the real fusion lives in the Worker (TS); this is an
# eval-side reimplementation over injectable text results.
# --------------------------------------------------------------------------
def rrf_fuse(
    image_products: Sequence[dict[str, Any]],
    text_products: Sequence[dict[str, Any]],
    *,
    k: int = DEFAULT_RRF_K,
    w_image: float = 1.0,
    w_text: float = 1.0,
    weak_visual_match: bool = False,
) -> list[str]:
    """Reciprocal-rank fusion of image + text lists -> ranked product ids.

    On ``weak_visual_match`` the image side is down-weighted so the ranking
    leans on text (mirrors the Worker fusion behaviour).
    """
    if weak_visual_match:
        w_image *= DEFAULT_WEAK_VISUAL_MULTIPLIER
    scores: dict[str, float] = {}
    for rank, hit in enumerate(image_products):
        pid = hit["product_id"]
        scores[pid] = scores.get(pid, 0.0) + w_image / (k + rank + 1)
    for hit in text_products:
        pid = hit["product_id"]
        rank = hit.get("rank", 0)
        scores[pid] = scores.get(pid, 0.0) + w_text / (k + rank + 1)
    return [pid for pid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))]


# --------------------------------------------------------------------------
# Indexing + query loop
# --------------------------------------------------------------------------
def _make_client(qdrant_url: Optional[str]):
    from qdrant_client import QdrantClient

    if qdrant_url:
        return QdrantClient(url=qdrant_url)
    return QdrantClient(":memory:")


def index_catalog(
    eval_set: "bes.EvalSet",
    dataset_dir: str,
    encoder,
    encoder_name: str,
    dim: int,
    client,
    collection_name: str,
) -> dict[str, Any]:
    """Embed + upsert all catalog reference images. Returns freshness info."""
    from PIL import Image

    settings = Settings(encoder="siglip2", embedding_dim_override=dim)
    qc.ensure_collection(settings, client=client, collection_name=collection_name, embedding_dim=dim)

    imgs, entries = [], []
    for e in eval_set.catalog:
        imgs.append(Image.open(os.path.join(dataset_dir, e.image_path)))
        entries.append(e)

    vectors = encoder.embed(imgs)
    indexed_at = time.time()
    points = [
        qc.build_point(
            vectors[i],
            image_id=e.image_id,
            product_id=e.product_id,
            encoder=encoder_name,
            embedding_dim=dim,
            is_reference=e.is_reference,
        )
        for i, e in enumerate(entries)
    ]
    qc.upsert(points, settings=settings, client=client, collection_name=collection_name)

    # Index freshness: indexed_at vs the newest catalog source_updated_at.
    src_times = [_parse_ts(e.source_updated_at) for e in entries if e.source_updated_at]
    src_times = [t for t in src_times if t is not None]
    freshness = {
        "indexed_at_epoch": indexed_at,
        "max_source_updated_at_epoch": max(src_times) if src_times else None,
        "freshness_lag_seconds": (indexed_at - max(src_times)) if src_times else None,
        "note": "D1 embedding_map not reachable in this env; indexed_at is the "
        "harness index time (freshness check is otherwise identical).",
    }
    return freshness


def _parse_ts(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def run_queries(
    eval_set: "bes.EvalSet",
    dataset_dir: str,
    encoder,
    dim: int,
    client,
    collection_name: str,
    text_provider: TextProvider,
    top_k: int,
    encoder_name: str,
) -> tuple[list[QueryResult], dict[str, LatencyStage]]:
    """Run every query through embed -> ANN -> group -> threshold (+ fusion)."""
    from PIL import Image

    settings = Settings(encoder="siglip2", embedding_dim_override=dim)
    lat = {name: LatencyStage() for name in ("embed", "ann", "fusion", "e2e")}
    results: list[QueryResult] = []

    for q in eval_set.queries:
        t0 = time.perf_counter()
        img = Image.open(os.path.join(dataset_dir, q.query_image_path))

        te = time.perf_counter()
        vec = encoder.embed([img])[0]
        lat["embed"].add((time.perf_counter() - te) * 1000)

        ta = time.perf_counter()
        hits = qc.query(vec, top_k=top_k, settings=settings, client=client, collection_name=collection_name)
        lat["ann"].add((time.perf_counter() - ta) * 1000)

        products = grouping.group_by_product(hits, pool="max")
        top_score = products[0]["score"] if products else 0.0
        weak = thr.apply_threshold(products, encoder=encoder_name)
        image_ranked = [p["product_id"] for p in products]

        text_hits = text_provider(q)
        text_ranked = [h["product_id"] for h in text_hits]

        tf = time.perf_counter()
        fused = rrf_fuse(products, text_hits, weak_visual_match=weak)
        lat["fusion"].add((time.perf_counter() - tf) * 1000)

        lat["e2e"].add((time.perf_counter() - t0) * 1000)
        results.append(
            QueryResult(
                query_image_path=q.query_image_path,
                expected_product_id=q.expected_product_id,
                in_catalog=q.in_catalog,
                ranked_products=image_ranked,
                top_score=top_score,
                weak_visual_match=weak,
                fused_ranked_products=fused,
                text_ranked_products=text_ranked,
            )
        )
    return results, lat


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _recall_at_k(results: Sequence[QueryResult], k: int, field_name: str) -> float:
    positives = [r for r in results if r.in_catalog and r.expected_product_id]
    if not positives:
        return 0.0
    hit = sum(
        1
        for r in positives
        if r.expected_product_id in getattr(r, field_name)[:k]
    )
    return hit / len(positives)


def compute_metrics(results: Sequence[QueryResult]) -> dict[str, Any]:
    positives = [r for r in results if r.in_catalog and r.expected_product_id]
    n_pos = len(positives)
    top1 = (
        sum(1 for r in positives if r.ranked_products[:1] == [r.expected_product_id]) / n_pos
        if n_pos
        else 0.0
    )
    weak_rate = sum(1 for r in results if r.weak_visual_match) / len(results) if results else 0.0
    return {
        "n_queries": len(results),
        "n_in_catalog": n_pos,
        "n_out_of_catalog": len(results) - n_pos,
        "recall@1": _recall_at_k(results, 1, "ranked_products"),
        "recall@5": _recall_at_k(results, 5, "ranked_products"),
        "recall@10": _recall_at_k(results, 10, "ranked_products"),
        "top1_accuracy": top1,
        "weak_match_rate": weak_rate,
    }


# --------------------------------------------------------------------------
# Threshold calibration
# --------------------------------------------------------------------------
def calibrate_threshold(
    results: Sequence[QueryResult],
    recall_target: float = DEFAULT_RECALL_TARGET,
    fp_limit: float = DEFAULT_FP_LIMIT,
) -> dict[str, Any]:
    """Derive a weak-match threshold from the eval set against explicit targets.

    For a candidate ``t`` a query is *accepted* (confident visual match) when
    ``top_score >= t``. We compute:
      * ``effective_recall@1`` = fraction of in-catalog queries whose top-1 is
        correct **and** accepted;
      * ``fpr`` = fraction of out-of-catalog queries that are accepted.

    We pick the highest ``t`` (lowest FPR) that keeps
    ``effective_recall@1 >= recall_target`` and ``fpr <= fp_limit``. If no
    candidate satisfies both, we pick the one minimising a shortfall cost and
    flag ``targets_met = False``.
    """
    positives = [r for r in results if r.in_catalog and r.expected_product_id]
    negatives = [r for r in results if not r.in_catalog]

    scores = sorted({round(r.top_score, 4) for r in results})
    # Candidate thresholds: midpoints between observed scores + a fine grid,
    # bounded to [0, 1] (cosine space).
    candidates = set(np.round(np.linspace(0.0, 1.0, 101), 4).tolist())
    for i in range(len(scores)):
        candidates.add(scores[i])
        if i + 1 < len(scores):
            candidates.add(round((scores[i] + scores[i + 1]) / 2, 4))
    ordered = sorted(candidates)

    def _accepted(r: QueryResult, t: float) -> bool:
        return r.top_score >= t

    rows = []
    for t in ordered:
        if positives:
            eff_recall = sum(
                1
                for r in positives
                if r.ranked_products[:1] == [r.expected_product_id] and _accepted(r, t)
            ) / len(positives)
        else:
            eff_recall = 0.0
        fpr = (
            sum(1 for r in negatives if _accepted(r, t)) / len(negatives)
            if negatives
            else 0.0
        )
        rows.append({"threshold": t, "effective_recall@1": eff_recall, "fpr": fpr})

    feasible = [
        row for row in rows
        if row["effective_recall@1"] >= recall_target and row["fpr"] <= fp_limit
    ]
    if feasible:
        # Highest threshold among feasible = lowest FPR while meeting recall.
        chosen = max(feasible, key=lambda r: r["threshold"])
        targets_met = True
    else:
        # Minimise combined shortfall (recall miss + fpr overage).
        chosen = min(
            rows,
            key=lambda r: max(0.0, recall_target - r["effective_recall@1"])
            + max(0.0, r["fpr"] - fp_limit),
        )
        targets_met = False

    return {
        "chosen_threshold": chosen["threshold"],
        "effective_recall@1_at_chosen": chosen["effective_recall@1"],
        "fpr_at_chosen": chosen["fpr"],
        "recall_target": recall_target,
        "fp_limit": fp_limit,
        "targets_met": targets_met,
        "n_positives": len(positives),
        "n_negatives": len(negatives),
    }


def write_threshold_file(
    path: str,
    encoder_name: str,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Merge the calibrated per-encoder threshold into a JSON config file.

    Existing entries for other encoders are preserved so the file accumulates
    re-calibrations per encoder (the runbook references this file).
    """
    data: dict[str, Any] = {"encoders": {}}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict) and isinstance(loaded.get("encoders"), dict):
                data = loaded
        except (json.JSONDecodeError, OSError):
            pass

    data.setdefault("encoders", {})
    data["encoders"][encoder_name] = {
        "weak_visual_match_threshold": calibration["chosen_threshold"],
        "recall_target": calibration["recall_target"],
        "fp_limit": calibration["fp_limit"],
        "effective_recall@1": calibration["effective_recall@1_at_chosen"],
        "fpr": calibration["fpr_at_chosen"],
        "targets_met": calibration["targets_met"],
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return data


# --------------------------------------------------------------------------
# A/B report: fused image+text vs text-only
# --------------------------------------------------------------------------
def ab_report(results: Sequence[QueryResult]) -> dict[str, Any]:
    fused = {
        "recall@1": _recall_at_k(results, 1, "fused_ranked_products"),
        "recall@5": _recall_at_k(results, 5, "fused_ranked_products"),
    }
    text_only = {
        "recall@1": _recall_at_k(results, 1, "text_ranked_products"),
        "recall@5": _recall_at_k(results, 5, "text_ranked_products"),
    }
    image_only = {
        "recall@1": _recall_at_k(results, 1, "ranked_products"),
        "recall@5": _recall_at_k(results, 5, "ranked_products"),
    }
    return {
        "fused_image_text": fused,
        "text_only": text_only,
        "image_only": image_only,
        "delta_fused_vs_text": {
            "recall@1": round(fused["recall@1"] - text_only["recall@1"], 4),
            "recall@5": round(fused["recall@5"] - text_only["recall@5"], 4),
        },
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_eval(
    dataset_dir: str,
    *,
    fake_encoder: bool = True,
    qdrant_url: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    recall_target: float = DEFAULT_RECALL_TARGET,
    fp_limit: float = DEFAULT_FP_LIMIT,
    threshold_out: Optional[str] = None,
    text_provider: Optional[TextProvider] = None,
    encoder_settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """Run the full eval and return a structured report dict."""
    eval_set = bes.load_manifests(dataset_dir)
    settings = encoder_settings or Settings()
    encoder, encoder_name, dim = build_encoder(fake_encoder, settings)

    client = _make_client(qdrant_url)
    collection_name = f"eval_{encoder_name}_{dim}"

    freshness = index_catalog(
        eval_set, dataset_dir, encoder, encoder_name, dim, client, collection_name
    )

    if text_provider is None:
        all_pids = sorted({e.product_id for e in eval_set.catalog})
        text_provider = default_mock_text_provider(all_pids)

    results, lat = run_queries(
        eval_set, dataset_dir, encoder, dim, client, collection_name,
        text_provider, top_k, encoder_name,
    )

    metrics = compute_metrics(results)
    latency = {name: stage.percentiles() for name, stage in lat.items()}
    calibration = calibrate_threshold(results, recall_target, fp_limit)
    ab = ab_report(results)

    threshold_path = threshold_out or os.path.join(_HERE, "thresholds.json")
    threshold_file = write_threshold_file(threshold_path, encoder_name, calibration)

    return {
        "dataset": dataset_dir,
        "encoder": encoder_name,
        "embedding_dim": dim,
        "top_k": top_k,
        "metrics": metrics,
        "latency_ms": latency,
        "index_freshness": freshness,
        "calibration": calibration,
        "threshold_file_path": threshold_path,
        "threshold_file": threshold_file,
        "ab_report": ab,
    }


# --------------------------------------------------------------------------
# Reporting (human-readable)
# --------------------------------------------------------------------------
def print_report(report: dict[str, Any]) -> None:
    m = report["metrics"]
    lat = report["latency_ms"]
    cal = report["calibration"]
    ab = report["ab_report"]

    print("=" * 68)
    print(f"IMAGE-SEARCH EVAL — encoder={report['encoder']} dim={report['embedding_dim']} "
          f"dataset={report['dataset']}")
    print("=" * 68)
    print("\n-- Retrieval metrics --")
    print(f"  queries={m['n_queries']} (in-catalog={m['n_in_catalog']}, "
          f"out-of-catalog={m['n_out_of_catalog']})")
    print(f"  recall@1 = {m['recall@1']:.3f}")
    print(f"  recall@5 = {m['recall@5']:.3f}")
    print(f"  recall@10= {m['recall@10']:.3f}")
    print(f"  top-1 accuracy = {m['top1_accuracy']:.3f}")
    print(f"  weak-match rate = {m['weak_match_rate']:.3f}")

    print("\n-- Per-stage latency (ms) --")
    for stage in ("embed", "ann", "fusion", "e2e"):
        p = lat[stage]
        print(f"  {stage:6s}  p50={p['p50']:.2f}  p90={p['p90']:.2f}  "
              f"p99={p['p99']:.2f}  mean={p['mean']:.2f}")

    fr = report["index_freshness"]
    lag = fr.get("freshness_lag_seconds")
    print("\n-- Index freshness --")
    if lag is None:
        print("  (no source_updated_at timestamps in catalog)")
    else:
        print(f"  freshness lag = {lag:.1f}s (indexed_at - max source_updated_at)")

    print("\n-- Threshold calibration --")
    print(f"  encoder '{report['encoder']}' -> weak_visual_match threshold = "
          f"{cal['chosen_threshold']}")
    print(f"  effective recall@1 @ chosen = {cal['effective_recall@1_at_chosen']:.3f} "
          f"(target >= {cal['recall_target']})")
    print(f"  FPR @ chosen = {cal['fpr_at_chosen']:.3f} (limit <= {cal['fp_limit']})")
    print(f"  targets met = {cal['targets_met']}")
    print(f"  written to: {report['threshold_file_path']}")
    print("  calibrated per-encoder thresholds:")
    for enc, cfg in sorted(report["threshold_file"].get("encoders", {}).items()):
        print(f"    {enc}: {cfg['weak_visual_match_threshold']} "
              f"(recall@1={cfg['effective_recall@1']:.3f}, fpr={cfg['fpr']:.3f}, "
              f"targets_met={cfg['targets_met']})")

    print("\n-- A/B: fused image+text vs text-only --")
    print(f"  image-only : recall@1={ab['image_only']['recall@1']:.3f}  "
          f"recall@5={ab['image_only']['recall@5']:.3f}")
    print(f"  text-only  : recall@1={ab['text_only']['recall@1']:.3f}  "
          f"recall@5={ab['text_only']['recall@5']:.3f}")
    print(f"  fused      : recall@1={ab['fused_image_text']['recall@1']:.3f}  "
          f"recall@5={ab['fused_image_text']['recall@5']:.3f}")
    d = ab["delta_fused_vs_text"]
    print(f"  delta fused-vs-text: recall@1={d['recall@1']:+.3f}  "
          f"recall@5={d['recall@5']:+.3f}")
    print("=" * 68)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Image-search eval harness (Task 10).")
    parser.add_argument("--dataset", default="eval/dataset", help="Path to labeled eval dataset dir.")
    parser.add_argument("--fake-encoder", action="store_true",
                        help="Use the deterministic fake encoder (no model download / GPU).")
    parser.add_argument("--qdrant-url", default=None,
                        help="Qdrant URL (e.g. http://localhost:6333). Default: in-memory.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--recall-target", type=float, default=DEFAULT_RECALL_TARGET)
    parser.add_argument("--fp-limit", type=float, default=DEFAULT_FP_LIMIT)
    parser.add_argument("--threshold-out", default=None,
                        help="Path to write calibrated per-encoder thresholds JSON. "
                             "Default: eval/thresholds.json.")
    parser.add_argument("--json", action="store_true", help="Also print the full JSON report.")
    args = parser.parse_args(argv)

    report = run_eval(
        args.dataset,
        fake_encoder=args.fake_encoder,
        qdrant_url=args.qdrant_url,
        top_k=args.top_k,
        recall_target=args.recall_target,
        fp_limit=args.fp_limit,
        threshold_out=args.threshold_out,
    )
    print_report(report)
    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
