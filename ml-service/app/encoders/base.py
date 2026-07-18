"""Encoder protocol + shared helpers (Task 4).

An ``Encoder`` turns a batch of PIL images into an L2-normalized float32
embedding matrix (cosine-ready) and reports its output dimension.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image


@runtime_checkable
class Encoder(Protocol):
    """Encoder interface. Implementations: dinov3.py, siglip2.py."""

    @property
    def dim(self) -> int:
        """Output embedding dimension."""
        ...

    def embed(self, images: "Sequence[Image]") -> "np.ndarray":
        """Return an (N, dim) float32, L2-normalized array for N images."""
        ...


def l2_normalize(matrix: "np.ndarray", eps: float = 1e-12) -> "np.ndarray":
    """Row-wise L2 normalization on a float32 (N, dim) matrix.

    Guards against zero vectors via ``eps`` so normalization never divides by
    zero. Output dtype is float32 (cosine-ready).
    """
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (arr / norms).astype(np.float32)
