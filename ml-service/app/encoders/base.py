"""Encoder protocol (Task 4 fills in implementations).

An ``Encoder`` turns a batch of PIL images into an L2-normalized float32
embedding matrix (cosine-ready) and reports its output dimension.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
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
