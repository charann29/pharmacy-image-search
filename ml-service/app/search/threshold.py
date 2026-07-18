"""Confidence thresholding + weak-match signal (Task 6; calibrated by Task 10).

Ships a safe provisional default per encoder. Task 10 (``eval/run_eval.py``)
calibrates per-encoder thresholds and writes them to a JSON file
(``{"encoders": {"<enc>": {"weak_visual_match_threshold": <float>, ...}}}``).
When ``WEAK_MATCH_THRESHOLD_FILE`` is configured and contains an entry for the
active encoder, that calibrated value is used; otherwise the provisional
default (0.35) applies.

When the top product's pooled score is below the active threshold,
``weak_visual_match`` is set so the Worker fusion layer can down-weight image
results and lean on text search.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Safe provisional default (cosine similarity space). Deliberately
# conservative so genuine out-of-catalog photos flag weak; Task 10 recalibrates
# per-encoder from the eval set.
DEFAULT_WEAK_MATCH_THRESHOLD = 0.35

# Per-encoder provisional overrides (all fall back to the default until Task 10
# writes calibrated values to WEAK_MATCH_THRESHOLD_FILE).
PER_ENCODER_THRESHOLDS: dict[str, float] = {
    "siglip2": 0.35,
    "dinov3": 0.35,
}


def load_calibrated_thresholds(path: Optional[str]) -> dict[str, float]:
    """Load per-encoder calibrated thresholds from the eval JSON file.

    Returns a ``{encoder: threshold}`` mapping. Missing/invalid file yields an
    empty mapping (caller falls back to the provisional default). Matches the
    schema written by ``eval/run_eval.py::write_threshold_file``.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read weak-match threshold file %s: %s", path, exc)
        return {}

    encoders = data.get("encoders") if isinstance(data, dict) else None
    if not isinstance(encoders, dict):
        return {}

    out: dict[str, float] = {}
    for enc, cfg in encoders.items():
        if isinstance(cfg, dict) and "weak_visual_match_threshold" in cfg:
            try:
                out[enc] = float(cfg["weak_visual_match_threshold"])
            except (TypeError, ValueError):
                continue
    return out


def threshold_for_encoder(
    encoder: Optional[str],
    calibrated: Optional[dict[str, float]] = None,
) -> float:
    """Return the weak-match threshold for an encoder.

    Precedence: calibrated value (if provided for this encoder) > provisional
    per-encoder default > global default.
    """
    if encoder and calibrated and encoder in calibrated:
        return calibrated[encoder]
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
    calibrated: Optional[dict[str, float]] = None,
) -> bool:
    """Compute the ``weak_visual_match`` flag for a ranked product list.

    Empty results are always weak. If ``threshold`` is None it is resolved from
    the encoder's calibrated value (when present in ``calibrated``) or the
    provisional default.
    """
    if threshold is None:
        threshold = threshold_for_encoder(encoder, calibrated)
    if not products:
        return True
    return is_weak_match(products[0]["score"], threshold)
