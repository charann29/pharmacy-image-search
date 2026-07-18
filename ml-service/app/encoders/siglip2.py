"""SigLIP2 encoder (STUB — implemented by Task 4).

Apache-2.0 launch encoder (fallback/default). Uses the HF SigLIP2 image tower.
Selected when ``ENCODER=siglip2``.
"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from ..config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from PIL.Image import Image

MODEL_ID = "google/siglip2-so400m-patch16-naflex"


class SigLIP2Encoder:
    """STUB: to be implemented by Task 4."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def dim(self) -> int:
        return self._settings.embedding_dim

    def embed(self, images: "Sequence[Image]") -> "np.ndarray":
        raise NotImplementedError("SigLIP2Encoder.embed is implemented by Task 4.")
