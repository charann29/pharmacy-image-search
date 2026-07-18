"""PaddleOCR 3.x wrapper + result normalizer (Task 5).

Wraps ``PaddleOCR(lang="en", use_textline_orientation=True).predict(image)`` and
normalizes the 3.x result object into internal ``{text, confidence, bbox}``
tokens. The 3.x result exposes parallel arrays ``rec_texts`` / ``rec_scores``
and polygons in ``rec_polys`` (4-point) or axis-aligned ``rec_boxes``; these
live either as attributes on the Result object, under ``res.json``, or in a
plain dict. :func:`normalize_result` handles all three.

``extract_name_strength`` picks a candidate product name + strength token
(e.g. "650 mg") by ranking lines on box area and confidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# PaddleOCR is heavy + GPU-bound; imported lazily inside the engine wrapper.

STRENGTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s?(mg|ml|mcg|g|iu)\b", re.IGNORECASE)


@dataclass
class OcrToken:
    text: str
    confidence: float
    bbox: list[list[float]] = field(default_factory=list)

    @property
    def area(self) -> float:
        """Axis-aligned bounding area of the polygon (0 if degenerate)."""
        if not self.bbox:
            return 0.0
        xs = [pt[0] for pt in self.bbox]
        ys = [pt[1] for pt in self.bbox]
        return float((max(xs) - min(xs)) * (max(ys) - min(ys)))


def _poly_from_box(box: Any) -> list[list[float]]:
    """Convert an axis-aligned [x1,y1,x2,y2] box to a 4-point polygon."""
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _poly_from_any(poly: Any) -> list[list[float]]:
    """Coerce a polygon-ish value into a list of [x, y] float pairs."""
    if poly is None:
        return []
    # numpy array -> list
    if hasattr(poly, "tolist"):
        poly = poly.tolist()
    # Flat [x1,y1,x2,y2] box.
    if len(poly) == 4 and all(isinstance(v, (int, float)) for v in poly):
        return _poly_from_box(poly)
    return [[float(pt[0]), float(pt[1])] for pt in poly]


def _extract_field(raw: Any, key: str) -> Any:
    """Read ``key`` from a Result object, its ``.json``, or a plain dict."""
    # Plain dict
    if isinstance(raw, dict):
        if key in raw:
            return raw[key]
        # PaddleX sometimes nests under 'res'.
        inner = raw.get("res")
        if isinstance(inner, dict) and key in inner:
            return inner[key]
        return None
    # Object with a .json property (PaddleOCR 3.x Result).
    js = getattr(raw, "json", None)
    if isinstance(js, dict):
        if key in js:
            return js[key]
        inner = js.get("res")
        if isinstance(inner, dict) and key in inner:
            return inner[key]
    # Direct attribute access as a last resort.
    return getattr(raw, key, None)


def normalize_result(raw: Any) -> list[OcrToken]:
    """Normalize one PaddleOCR 3.x result into a list of OcrToken.

    Accepts a single Result object/dict, or a list of them (predict returns a
    list). Missing polygons degrade to an empty bbox rather than failing.
    """
    # predict() returns a list; accept either a list or a single result.
    if isinstance(raw, (list, tuple)) and not (
        len(raw) and isinstance(raw[0], (int, float))
    ):
        tokens: list[OcrToken] = []
        for item in raw:
            tokens.extend(normalize_result(item))
        return tokens

    texts = _extract_field(raw, "rec_texts") or []
    scores = _extract_field(raw, "rec_scores") or []
    polys = _extract_field(raw, "rec_polys")
    if not polys:
        polys = _extract_field(raw, "rec_boxes")

    tokens = []
    n = len(texts)
    for i in range(n):
        text = str(texts[i]).strip()
        conf = float(scores[i]) if i < len(scores) else 0.0
        bbox: list[list[float]] = []
        if polys is not None and i < len(polys):
            bbox = _poly_from_any(polys[i])
        if text:
            tokens.append(OcrToken(text=text, confidence=conf, bbox=bbox))
    return tokens


def _parse_strength(text: str) -> Optional[str]:
    """Return a normalized strength token like '650 mg' if present."""
    m = STRENGTH_RE.search(text)
    if not m:
        return None
    return f"{m.group(1)} {m.group(2).lower()}"


def extract_name_strength(tokens: list[OcrToken]) -> dict[str, Any]:
    """Extract a candidate product name + strength from OCR tokens.

    Ranking heuristic: score each line by box area and confidence. The
    highest-ranked line that is not purely a strength/number is the candidate
    name; the first strength token found (ranked the same way) is the strength.

    Returns ``{name, strength, candidates: [str], tokens: [{...}]}``. Empty
    inputs return empty candidates gracefully.
    """
    if not tokens:
        return {"name": None, "strength": None, "candidates": [], "tokens": []}

    max_area = max((t.area for t in tokens), default=0.0) or 1.0

    def rank(t: OcrToken) -> float:
        # Normalize area to [0,1] then blend with confidence.
        area_score = t.area / max_area
        return 0.6 * area_score + 0.4 * t.confidence

    ranked = sorted(tokens, key=rank, reverse=True)

    strength: Optional[str] = None
    for t in ranked:
        s = _parse_strength(t.text)
        if s:
            strength = s
            break

    name: Optional[str] = None
    for t in ranked:
        cleaned = t.text.strip()
        # Skip lines that are just a strength/number.
        if not cleaned:
            continue
        if _parse_strength(cleaned) and not re.search(r"[A-Za-z]{3,}", cleaned):
            continue
        if re.search(r"[A-Za-z]{2,}", cleaned):
            name = cleaned
            break

    # Candidate query strings for the existing text search, best-first.
    candidates: list[str] = []
    if name and strength:
        candidates.append(f"{name} {strength}")
    if name:
        candidates.append(name)
    # Include the top few high-signal lines as extra query candidates.
    for t in ranked[:3]:
        if t.text not in candidates:
            candidates.append(t.text)

    return {
        "name": name,
        "strength": strength,
        "candidates": candidates,
        "tokens": [
            {"text": t.text, "confidence": t.confidence, "bbox": t.bbox}
            for t in tokens
        ],
    }


class PaddleOcrEngine:
    """Lazy-loading PaddleOCR 3.x engine wrapper."""

    def __init__(self, lang: str = "en", use_textline_orientation: bool = True) -> None:
        self._lang = lang
        self._use_textline_orientation = use_textline_orientation
        self._ocr = None

    def _ensure_loaded(self) -> None:
        if self._ocr is not None:
            return
        from paddleocr import PaddleOCR  # noqa: WPS433

        self._ocr = PaddleOCR(
            lang=self._lang,
            use_textline_orientation=self._use_textline_orientation,
        )

    def run(self, image: Any) -> dict[str, Any]:
        """Run OCR on an image (path or ndarray) and return extracted fields."""
        self._ensure_loaded()
        raw = self._ocr.predict(image)
        tokens = normalize_result(raw)
        return extract_name_strength(tokens)


_ENGINE: Optional[PaddleOcrEngine] = None


def get_engine() -> PaddleOcrEngine:
    """Process-cached PaddleOCR engine."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PaddleOcrEngine()
    return _ENGINE
