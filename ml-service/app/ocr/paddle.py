"""PaddleOCR 3.x wrapper + result normalizer (STUB — implemented by Task 5).

Wraps ``PaddleOCR(lang="en", use_textline_orientation=True).predict(image)`` and
normalizes the 3.x result object (``rec_texts``, ``rec_scores``,
``rec_polys``/``rec_boxes``) into ``{text, confidence, bbox}`` tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OcrToken:
    text: str
    confidence: float
    bbox: list[list[float]] = field(default_factory=list)


def normalize_result(raw: Any) -> list[OcrToken]:
    """Normalize a PaddleOCR 3.x result object into OcrToken list.

    STUB: implemented by Task 5.
    """
    raise NotImplementedError("normalize_result is implemented by Task 5.")


def extract_name_strength(tokens: list[OcrToken]) -> dict[str, Any]:
    """Extract candidate product name + strength (e.g. '650 mg').

    STUB: implemented by Task 5.
    """
    raise NotImplementedError("extract_name_strength is implemented by Task 5.")
