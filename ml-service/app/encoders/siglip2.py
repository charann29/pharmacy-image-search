"""SigLIP2 encoder (Task 4) — Apache-2.0 launch/default encoder.

Uses the HF SigLIP2 image tower via ``AutoModel``/``AutoProcessor`` and
``get_image_features``. Selected when ``ENCODER=siglip2``. Runs under
``torch.inference_mode()``, fp16 on GPU, GPU batching, and returns
L2-normalized float32 embeddings. Lazy-loads the model on first ``embed()``.
"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import numpy as np

from ..config import Settings, get_settings
from .base import l2_normalize
from .preprocess import preprocess_batch

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

MODEL_ID = "google/siglip2-so400m-patch16-naflex"


class SigLIP2Encoder:
    """SigLIP2 image-tower encoder. Lazy-loads the HF model + processor."""

    model_id = MODEL_ID

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None

    @property
    def dim(self) -> int:
        return self._settings.embedding_dim

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: WPS433
        from transformers import AutoModel, AutoProcessor  # noqa: WPS433

        self._torch = torch
        device = self._settings.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self._device = device

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModel.from_pretrained(self.model_id)
        model.eval()
        model.to(device)
        if self._settings.use_fp16 and device.startswith("cuda"):
            model.half()
        self._model = model

    def embed(self, images: "Sequence[Image]") -> "np.ndarray":
        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        torch = self._torch

        processed = preprocess_batch(list(images))
        batch_size = self._settings.batch_size
        chunks: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(processed), batch_size):
                batch = processed[start : start + batch_size]
                inputs = self._processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                if self._settings.use_fp16 and self._device.startswith("cuda"):
                    inputs = {
                        k: (v.half() if v.is_floating_point() else v)
                        for k, v in inputs.items()
                    }
                features = self._model.get_image_features(**inputs)
                chunks.append(features.float().cpu().numpy())

        embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
        return l2_normalize(embeddings)
