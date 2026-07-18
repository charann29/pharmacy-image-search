"""Tests for the eval harness (Task 10).

Runs the full harness on the committed synthetic fixture with the deterministic
fake encoder + in-memory Qdrant, and asserts metrics are computed and a
per-encoder threshold file is written. Also covers the builder round-trip, RRF
fusion, and threshold calibration against explicit targets.

These tests need numpy + qdrant-client + pillow + pydantic-settings (all in
ml-service/requirements.txt). They skip cleanly if a dep is missing.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.dirname(_HERE)
for _p in (_EVAL_DIR, os.path.join(os.path.dirname(_EVAL_DIR), "ml-service")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


pytestmark = pytest.mark.skipif(
    not (_have("qdrant_client") and _have("PIL") and _have("pydantic_settings") and _have("numpy")),
    reason="requires qdrant-client, pillow, pydantic-settings, numpy",
)

DATASET = os.path.join(_EVAL_DIR, "dataset")


@pytest.fixture(scope="module")
def modules():
    import build_eval_set as bes
    import run_eval as re

    return bes, re


# --------------------------------------------------------------------------
# build_eval_set
# --------------------------------------------------------------------------
def test_fixture_exists_and_loads(modules):
    bes, _ = modules
    assert os.path.isdir(DATASET), "sample dataset fixture must be committed"
    es = bes.load_manifests(DATASET)
    assert es.catalog, "catalog manifest must be non-empty"
    assert es.queries, "queries manifest must be non-empty"
    # Both positives and out-of-catalog negatives present (needed for FPR calc).
    assert any(q.in_catalog for q in es.queries)
    assert any(not q.in_catalog for q in es.queries)
    # Every catalog image path resolves on disk.
    for e in es.catalog:
        assert os.path.exists(os.path.join(DATASET, e.image_path))


def test_generate_synthetic_roundtrip(modules, tmp_path):
    bes, _ = modules
    out = str(tmp_path / "ds")
    es = bes.generate_synthetic_dataset(out, n_products=3, catalog_per_product=2,
                                        queries_per_product=1, out_of_catalog=2)
    assert len(es.catalog) == 6
    assert len(es.queries) == 5  # 3 positives + 2 negatives
    loaded = bes.load_manifests(out)
    assert len(loaded.catalog) == len(es.catalog)
    assert os.path.exists(os.path.join(out, "README.md"))
    assert os.path.exists(os.path.join(out, bes.CATALOG_MANIFEST))
    assert os.path.exists(os.path.join(out, bes.QUERIES_MANIFEST))


def test_assemble_from_rows_skips_unmapped(modules):
    bes, _ = modules
    catalog_rows = [
        {"product_id": "P1", "image_id": "i1", "image_path": "a.png"},
        {"product_id": "", "image_id": "i2", "image_path": "b.png"},  # unmapped -> skipped
    ]
    query_rows = [{"query_image_path": "q1.png", "expected_product_id": "P1"}]
    es = bes.assemble_from_rows(catalog_rows, query_rows)
    assert len(es.catalog) == 1
    assert es.catalog[0].product_id == "P1"


# --------------------------------------------------------------------------
# Full harness on the fixture
# --------------------------------------------------------------------------
def test_run_eval_end_to_end_writes_thresholds(modules, tmp_path):
    _, re = modules
    thr_path = str(tmp_path / "thresholds.json")
    report = re.run_eval(DATASET, fake_encoder=True, threshold_out=thr_path)

    # Metrics computed.
    m = report["metrics"]
    for key in ("recall@1", "recall@5", "recall@10", "top1_accuracy", "weak_match_rate"):
        assert key in m
        assert 0.0 <= m[key] <= 1.0
    # Fake encoder cleanly separates products -> perfect recall on the fixture.
    assert m["recall@1"] == 1.0
    assert m["n_in_catalog"] > 0 and m["n_out_of_catalog"] > 0

    # Latency stages present with percentiles.
    for stage in ("embed", "ann", "fusion", "e2e"):
        assert set(report["latency_ms"][stage]) >= {"p50", "p90", "p99", "mean"}

    # Index freshness present.
    assert "freshness_lag_seconds" in report["index_freshness"]

    # Threshold file written + parseable + contains the encoder entry.
    assert os.path.exists(thr_path)
    with open(thr_path) as fh:
        data = json.load(fh)
    assert report["encoder"] in data["encoders"]
    entry = data["encoders"][report["encoder"]]
    assert 0.0 <= entry["weak_visual_match_threshold"] <= 1.0
    assert "targets_met" in entry

    # A/B report has fused-vs-text deltas.
    ab = report["ab_report"]
    assert "delta_fused_vs_text" in ab
    assert "recall@1" in ab["delta_fused_vs_text"]


def test_threshold_file_accumulates_per_encoder(modules, tmp_path):
    _, re = modules
    thr_path = str(tmp_path / "thresholds.json")
    # First encoder.
    re.write_threshold_file(thr_path, "siglip2", {
        "chosen_threshold": 0.4, "recall_target": 0.8, "fp_limit": 0.2,
        "effective_recall@1_at_chosen": 0.9, "fpr_at_chosen": 0.1, "targets_met": True,
    })
    # Second encoder must not clobber the first.
    re.write_threshold_file(thr_path, "dinov3", {
        "chosen_threshold": 0.5, "recall_target": 0.8, "fp_limit": 0.2,
        "effective_recall@1_at_chosen": 0.85, "fpr_at_chosen": 0.05, "targets_met": True,
    })
    with open(thr_path) as fh:
        data = json.load(fh)
    assert set(data["encoders"]) == {"siglip2", "dinov3"}


# --------------------------------------------------------------------------
# Calibration + fusion units
# --------------------------------------------------------------------------
def test_calibrate_threshold_respects_targets(modules):
    _, re = modules
    QR = re.QueryResult
    # 4 positives correctly ranked at high score; 2 negatives at low score.
    results = [
        QR("p0", "P0", True, ["P0"], 0.90, False, ["P0"], ["P0"]),
        QR("p1", "P1", True, ["P1"], 0.88, False, ["P1"], ["P1"]),
        QR("p2", "P2", True, ["P2"], 0.85, False, ["P2"], ["P2"]),
        QR("p3", "P3", True, ["P3"], 0.83, False, ["P3"], ["P3"]),
        QR("n0", None, False, ["P9"], 0.30, True, ["P9"], []),
        QR("n1", None, False, ["P8"], 0.25, True, ["P8"], []),
    ]
    cal = re.calibrate_threshold(results, recall_target=0.8, fp_limit=0.2)
    assert cal["targets_met"] is True
    # Chosen threshold separates positives (>=0.83) from negatives (<=0.30).
    assert 0.30 < cal["chosen_threshold"] <= 0.83
    assert cal["fpr_at_chosen"] == 0.0


def test_rrf_fuse_weak_match_leans_on_text(modules):
    _, re = modules
    image = [{"product_id": "IMG_TOP", "score": 0.9}, {"product_id": "P1", "score": 0.8}]
    text = [{"product_id": "P1", "score": 1.0, "rank": 0}, {"product_id": "P2", "score": 0.9, "rank": 1}]
    # Strong visual match keeps image top ranked highly.
    strong = re.rrf_fuse(image, text, weak_visual_match=False)
    # Weak visual match down-weights image so text's P1 wins.
    weak = re.rrf_fuse(image, text, weak_visual_match=True)
    assert weak[0] == "P1"
    assert "IMG_TOP" in strong


def test_injectable_text_provider(modules):
    _, re = modules
    QE = __import__("build_eval_set").QueryEntry

    def provider(q):
        return [{"product_id": "CUSTOM", "score": 1.0, "rank": 0}]

    report = re.run_eval(DATASET, fake_encoder=True,
                         threshold_out=os.path.join(_HERE, "_tmp_thr.json"),
                         text_provider=provider)
    # Text-only recall should be 0 since provider never returns the expected id.
    assert report["ab_report"]["text_only"]["recall@1"] == 0.0
    os.remove(os.path.join(_HERE, "_tmp_thr.json"))
