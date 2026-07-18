"""Tests for search grouping (Task 6) and weak-match thresholding."""
from __future__ import annotations

import pytest

from app.search.grouping import group_by_product
from app.search.threshold import (
    DEFAULT_WEAK_MATCH_THRESHOLD,
    apply_threshold,
    is_weak_match,
    threshold_for_encoder,
)


def _hit(product_id, score):
    return {"score": score, "payload": {"product_id": product_id, "image_id": f"{product_id}-img"}}


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------
def test_group_dedupes_four_images_to_one_product():
    hits = [_hit("p1", s) for s in (0.9, 0.85, 0.7, 0.6)]
    grouped = group_by_product(hits, pool="max")
    assert len(grouped) == 1
    assert grouped[0]["product_id"] == "p1"
    assert grouped[0]["score"] == 0.9  # max pool
    assert grouped[0]["image_count"] == 4


def test_group_mean_pool():
    hits = [_hit("p1", 0.8), _hit("p1", 0.4)]
    grouped = group_by_product(hits, pool="mean")
    assert grouped[0]["score"] == pytest.approx(0.6)


def test_group_ranks_products_by_pooled_score():
    hits = [_hit("p1", 0.5), _hit("p2", 0.9), _hit("p3", 0.7)]
    grouped = group_by_product(hits, pool="max")
    assert [g["product_id"] for g in grouped] == ["p2", "p3", "p1"]


def test_group_supports_top_level_product_id():
    hits = [{"score": 0.9, "product_id": "p1"}, {"score": 0.8, "product_id": "p1"}]
    grouped = group_by_product(hits, pool="max")
    assert grouped[0]["image_count"] == 2


def test_group_skips_hits_without_product_id():
    hits = [{"score": 0.9, "payload": {}}, _hit("p1", 0.5)]
    grouped = group_by_product(hits)
    assert [g["product_id"] for g in grouped] == ["p1"]


def test_group_tie_break_deterministic():
    hits = [_hit("pb", 0.9), _hit("pa", 0.9)]
    grouped = group_by_product(hits)
    assert [g["product_id"] for g in grouped] == ["pa", "pb"]


def test_group_invalid_pool_raises():
    with pytest.raises(ValueError):
        group_by_product([_hit("p1", 0.5)], pool="median")


def test_group_empty():
    assert group_by_product([]) == []


# --------------------------------------------------------------------------
# threshold
# --------------------------------------------------------------------------
def test_is_weak_match_below_threshold():
    assert is_weak_match(0.2, 0.35) is True
    assert is_weak_match(0.5, 0.35) is False


def test_is_weak_match_boundary():
    # Exactly at threshold is NOT weak (strict less-than).
    assert is_weak_match(0.35, 0.35) is False


def test_threshold_for_encoder_defaults():
    assert threshold_for_encoder("siglip2") == 0.35
    assert threshold_for_encoder("dinov3") == 0.35
    assert threshold_for_encoder("unknown") == DEFAULT_WEAK_MATCH_THRESHOLD
    assert threshold_for_encoder(None) == DEFAULT_WEAK_MATCH_THRESHOLD


def test_apply_threshold_strong_match():
    products = [{"product_id": "p1", "score": 0.9}]
    assert apply_threshold(products, encoder="siglip2") is False


def test_apply_threshold_weak_match():
    products = [{"product_id": "p1", "score": 0.1}]
    assert apply_threshold(products, encoder="siglip2") is True


def test_apply_threshold_empty_is_weak():
    assert apply_threshold([], encoder="siglip2") is True


def test_apply_threshold_explicit_threshold_overrides_encoder():
    products = [{"product_id": "p1", "score": 0.5}]
    assert apply_threshold(products, threshold=0.6) is True
    assert apply_threshold(products, threshold=0.4) is False
