"""Tests for the OCR normalizer + name/strength extraction (Task 5).

Runs against fabricated PaddleOCR 3.x result fixtures (Result-object-with-.json,
plain dict, and rec_boxes variants) — no PaddleOCR install / model needed.
"""
from __future__ import annotations

from app.ocr.paddle import (
    OcrToken,
    extract_name_strength,
    normalize_result,
)


class _FakeResult:
    """Stand-in for a PaddleOCR 3.x Result object exposing ``.json``."""

    def __init__(self, json_dict):
        self.json = json_dict


def _box_fixture():
    """A fabricated medicine-box OCR result: 'Crocin' big, '650 mg' below."""
    return {
        "rec_texts": ["Crocin", "650 mg", "Paracetamol Tablets"],
        "rec_scores": [0.99, 0.97, 0.88],
        "rec_polys": [
            [[10, 10], [210, 10], [210, 70], [10, 70]],   # big -> name
            [[10, 80], [110, 80], [110, 110], [10, 110]],  # strength
            [[10, 120], [180, 120], [180, 140], [10, 140]],
        ],
    }


def test_normalize_result_object_with_json():
    tokens = normalize_result(_FakeResult(_box_fixture()))
    assert len(tokens) == 3
    assert tokens[0].text == "Crocin"
    assert abs(tokens[0].confidence - 0.99) < 1e-6
    assert tokens[0].bbox[0] == [10.0, 10.0]


def test_normalize_result_plain_dict():
    tokens = normalize_result(_box_fixture())
    assert [t.text for t in tokens] == ["Crocin", "650 mg", "Paracetamol Tablets"]


def test_normalize_result_list_of_results():
    tokens = normalize_result([_FakeResult(_box_fixture())])
    assert len(tokens) == 3


def test_normalize_result_rec_boxes_axis_aligned():
    fixture = {
        "rec_texts": ["Aspirin"],
        "rec_scores": [0.9],
        "rec_boxes": [[5, 5, 105, 45]],  # x1,y1,x2,y2
    }
    tokens = normalize_result(fixture)
    assert len(tokens) == 1
    # Converted to a 4-point polygon.
    assert tokens[0].bbox == [[5.0, 5.0], [105.0, 5.0], [105.0, 45.0], [5.0, 45.0]]


def test_normalize_result_skips_empty_text():
    fixture = {"rec_texts": ["", "  ", "Ok"], "rec_scores": [0.5, 0.5, 0.9], "rec_polys": [None, None, None]}
    tokens = normalize_result(fixture)
    assert [t.text for t in tokens] == ["Ok"]


def test_extract_name_strength_basic():
    tokens = normalize_result(_box_fixture())
    result = extract_name_strength(tokens)
    assert result["name"] == "Crocin"
    assert result["strength"] == "650 mg"
    assert "Crocin 650 mg" in result["candidates"]


def test_extract_name_strength_variants():
    for raw, expected in [("500mg", "500 mg"), ("5 ml", "5 ml"), ("250 mcg", "250 mcg"), ("2g", "2 g"), ("100 IU", "100 iu")]:
        tokens = [OcrToken(text=f"DrugX {raw}", confidence=0.9, bbox=[[0, 0], [100, 0], [100, 30], [0, 30]])]
        result = extract_name_strength(tokens)
        assert result["strength"] == expected


def test_extract_name_strength_empty_graceful():
    result = extract_name_strength([])
    assert result["name"] is None
    assert result["strength"] is None
    assert result["candidates"] == []


def test_extract_name_strength_no_text_image():
    # No tokens (no-text image) -> empty candidates, no crash.
    result = extract_name_strength(normalize_result({"rec_texts": [], "rec_scores": [], "rec_polys": []}))
    assert result["candidates"] == []


def test_ranking_prefers_larger_box():
    # Small high-confidence noise vs large medium-confidence brand name.
    tokens = [
        OcrToken(text="tiny", confidence=0.99, bbox=[[0, 0], [10, 0], [10, 10], [0, 10]]),
        OcrToken(text="BigBrand", confidence=0.8, bbox=[[0, 0], [300, 0], [300, 80], [0, 80]]),
    ]
    result = extract_name_strength(tokens)
    assert result["name"] == "BigBrand"
