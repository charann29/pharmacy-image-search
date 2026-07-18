"""DINOv3 encoder (STUB — implemented by Task 4).

Loads ``facebook/dinov3-vitb16-pretrain-lvd1689m`` via HuggingFace Transformers
(gated access, see Task 0). Uses ``torch.inference_mode()``, fp16 on GPU, and
returns L2-normalized ``pooler_output`` (or mean of ``last_hidden_state``).
"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from ..config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from PIL.Image import Image

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class DinoV3Encoder:
    """STUB: to be implemented by Task 4."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def dim(self) -> int:
        return self._settings.embedding_dim

    def embed(self, images: "Sequence[Image]") -> "np.ndarray":
        raise NotImplementedError("DinoV3Encoder.embed is implemented by Task 4.")
