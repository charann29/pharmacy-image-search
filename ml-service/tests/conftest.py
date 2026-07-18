"""Shared pytest fixtures + import-availability guards for ml-service tests.

Tests are structured so the maximum runs without GPU/model-download/live
Qdrant. Heavy dependencies (torch/transformers, paddleocr) are mocked; Qdrant
uses in-memory mode when the client is installed. Real-model / GPU tests are
guarded by skip markers.
"""
from __future__ import annotations

import importlib.util

import pytest


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


HAVE_QDRANT = _have("qdrant_client")
HAVE_TORCH = _have("torch")
HAVE_TRANSFORMERS = _have("transformers")
HAVE_PADDLE = _have("paddleocr")
HAVE_PIL = _have("PIL")
HAVE_NUMPY = _have("numpy")

requires_qdrant = pytest.mark.skipif(not HAVE_QDRANT, reason="qdrant-client not installed")
requires_model = pytest.mark.skipif(
    not (HAVE_TORCH and HAVE_TRANSFORMERS),
    reason="torch/transformers not installed (model download env)",
)
requires_paddle = pytest.mark.skipif(not HAVE_PADDLE, reason="paddleocr not installed")


@pytest.fixture
def make_pil_image():
    """Factory for small solid/patterned RGB PIL images."""
    from PIL import Image

    def _make(color=(128, 64, 32), size=(64, 64)):
        return Image.new("RGB", size, color)

    return _make
