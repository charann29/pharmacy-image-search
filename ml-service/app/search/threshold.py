"""Confidence thresholding + weak-match signal (Task 6; calibrated by Task 10).

Ships a safe provisional default per encoder; Task 10 replaces these with
values calibrated against the eval set. When the top product's pooled score is
below the active threshold, ``weak_visual_match`` is set so the Worker fusion
layer can down-weight image results and lean on text search.
"""
from __future__ import annotations

from typing import Any, Optional

# Safe provisional default (cosine similarity space). Deliberately
# conservative so genuine out-of-catalog photos flag weak; Task 10 recalibrates
# per-encoder from the eval set.
DEFAULT_WEAK_MATCH_THRESHOLD = 0.35

# Per-encoder provisional overrides (all fall back to the default until Task 10
# writes calibrated values here / in config).
PER_ENCODER_THRESHOLDS: dict[str, float] = {
    "siglip2": 0.35,
    "dinov3": 0.35,
}


def threshold_for_encoder(encoder: Optional[str]) -> float:
    """Return the provisional weak-match threshold for an encoder."""
    if not encoder:
        return DEFAULT_WEAK_MATCH_THRESHOLD
    return PER_ENCODER_THRESHOLDS.get(encoder, DEFAULT_WEAK_MATCH_THRESHOLD)


def is_weak_match(
    top_score: float,
    threshold: float = DEFAULT_WEAK_MATCH_THRESHOLD,
) -> bool:
    """True when the top product score is below the weak-match threshold."""
    return float(top_score) < float(threshold)


def apply_threshold(
    products: list[dict[str, Any]],
    threshold: Optional[float] = None,
    encoder: Optional[str] = None,
) -> bool:
    """Compute the ``weak_visual_match`` flag for a ranked product list.

    Empty results are always weak. If ``threshold`` is None it is resolved from
    the encoder's provisional value.
    """
    if threshold is None:
        threshold = threshold_for_encoder(encoder)
    if not products:
        return True
    return is_weak_match(products[0]["score"], threshold)
