"""DINOv3 encoder (Task 4).

Loads ``facebook/dinov3-vitb16-pretrain-lvd1689m`` via HuggingFace Transformers
(gated access — see plan Task 0). Uses ``pooler_output`` (or mean of
``last_hidden_state``). Shared load/batch/normalize machinery lives in
``_BaseHFEncoder``; this class only supplies the two model-specific hooks.
"""
from __future__ import annotations

import numpy as np

from ._hf import _BaseHFEncoder

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class DinoV3Encoder(_BaseHFEncoder):
    """DINOv3 image encoder. Lazy-loads the HF model + processor on first use."""

    model_id = MODEL_ID

    def _load(self, model_id, device):
        from transformers import AutoImageProcessor, AutoModel  # noqa: WPS433

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.eval()
        model.to(device)
        if self._settings.use_fp16 and device.startswith("cuda"):
            model.half()
        return model, processor

    def _forward(self, model, processor, batch) -> "np.ndarray":
        inputs = self._prepare_inputs(batch)
        outputs = model(**inputs)
        # Prefer pooler_output; fall back to mean of last_hidden_state.
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state.mean(dim=1)
        return pooled.float().cpu().numpy()
