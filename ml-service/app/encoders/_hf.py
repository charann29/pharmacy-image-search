"""Shared HuggingFace encoder base (SIMPLIFY #1).

``_BaseHFEncoder`` holds everything DINOv3 and SigLIP2 share: construction,
``dim``, lazy model loading, and the batched ``embed()`` loop with
preprocessing, fp16/device handling, and L2 normalization. Concrete encoders
override two hooks:

    _load(model_id, device) -> (model, processor)
    _forward(model, processor, batch) -> np.ndarray  # (B, dim), pre-normalize
"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import numpy as np

from ..config import Settings, get_settings
from .base import l2_normalize
from .preprocess import preprocess_batch

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image


class _BaseHFEncoder:
    """Base class for HF image-tower encoders (lazy load + batched embed)."""

    model_id: str = ""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None

    @property
    def dim(self) -> int:
        return self._settings.embedding_dim

    # ---- Overridable hooks ------------------------------------------------
    def _load(self, model_id: str, device: str):
        """Return ``(model, processor)`` loaded from ``model_id``."""
        raise NotImplementedError

    def _forward(self, model, processor, batch: "list[Image]") -> "np.ndarray":
        """Run a preprocessed image batch through the model -> (B, dim) array."""
        raise NotImplementedError

    # ---- Shared machinery -------------------------------------------------
    def _resolve_device(self) -> str:
        import torch  # noqa: WPS433

        self._torch = torch
        device = self._settings.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        return device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        device = self._resolve_device()
        self._device = device
        model, processor = self._load(self.model_id, device)
        self._model = model
        self._processor = processor

    def _prepare_inputs(self, batch: "list[Image]"):
        """Processor -> device tensors (+ fp16 on floating inputs on GPU)."""
        inputs = self._processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        if self._settings.use_fp16 and self._device.startswith("cuda"):
            inputs = {
                k: (v.half() if v.is_floating_point() else v)
                for k, v in inputs.items()
            }
        return inputs

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
                chunks.append(self._forward(self._model, self._processor, batch))

        embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
        return l2_normalize(embeddings)
