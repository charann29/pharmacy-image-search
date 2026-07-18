"""SigLIP2 encoder (Task 4) — Apache-2.0 launch/default encoder.

Uses the HF SigLIP2 image tower via ``AutoModel``/``AutoProcessor`` and
``get_image_features``. Selected when ``ENCODER=siglip2``. Shared
load/batch/normalize machinery lives in ``_BaseHFEncoder``; this class only
supplies the two model-specific hooks.
"""
from __future__ import annotations

import numpy as np

from ._hf import _BaseHFEncoder

MODEL_ID = "google/siglip2-so400m-patch16-naflex"


class SigLIP2Encoder(_BaseHFEncoder):
    """SigLIP2 image-tower encoder. Lazy-loads the HF model + processor."""

    model_id = MODEL_ID

    def _load(self, model_id, device):
        from transformers import AutoModel, AutoProcessor  # noqa: WPS433

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.eval()
        model.to(device)
        if self._settings.use_fp16 and device.startswith("cuda"):
            model.half()
        return model, processor

    def _forward(self, model, processor, batch) -> "np.ndarray":
        inputs = self._prepare_inputs(batch)
        features = model.get_image_features(**inputs)
        # Older transformers return a tensor; newer (>=5.x) may return a
        # ModelOutput wrapper whose image embedding is ``pooler_output``.
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        elif not hasattr(features, "float"):
            features = features[0]
        return features.float().cpu().numpy()
