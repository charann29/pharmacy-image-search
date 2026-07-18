"""Confidence thresholding + weak-match signal (STUB — implemented by Task 6/10).

Ships a safe provisional default; Task 10 replaces with calibrated per-encoder
values.
"""
from __future__ import annotations

# Safe provisional default; recalibrated per-encoder by Task 10.
DEFAULT_WEAK_MATCH_THRESHOLD = 0.35


def is_weak_match(top_score: float, threshold: float = DEFAULT_WEAK_MATCH_THRESHOLD) -> bool:
    """True when the top product score is below the weak-match threshold."""
    return top_score < threshold
