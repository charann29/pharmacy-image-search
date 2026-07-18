"""Tests for encoders: base helpers, preprocessing, factory, and embed().

The HF models may not download in CI, so the encoder embed path is exercised
with a mocked torch/transformers stack that returns a fixed pooled tensor. One
real-shape test is guarded (skipped) when torch/transformers are unavailable.
Assertions cover: output shape == EMBEDDING_DIM, L2 norm ≈ 1.0, determinism,
and identical-image cosine ≈ 1.0.
"""
from __future__ import annotations

import sys
import types
from unittest import mock

import numpy as np
import pytest

from app.config import Settings
from app.encoders import get_encoder, reset_encoder_cache
from app.encoders.base import Encoder, l2_normalize
from app.encoders.preprocess import (
    auto_orient,
    center_crop,
    preprocess,
    resize_short_side,
    to_rgb,
)

from conftest import requires_model


# --------------------------------------------------------------------------
# base.l2_normalize
# --------------------------------------------------------------------------
def test_l2_normalize_unit_norm():
    m = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = l2_normalize(m)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert out.dtype == np.float32


def test_l2_normalize_handles_zero_vector():
    m = np.zeros((1, 4), dtype=np.float32)
    out = l2_normalize(m)  # must not divide by zero
    assert out.shape == (1, 4)
    assert np.all(np.isfinite(out))


def test_l2_normalize_1d_promoted():
    out = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert out.shape == (1, 2)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


# --------------------------------------------------------------------------
# preprocessing (Task 9)
# --------------------------------------------------------------------------
def test_preprocess_output_shape(make_pil_image):
    img = make_pil_image(size=(400, 300))
    out = preprocess(img, short_side=256, crop=224)
    assert out.size == (224, 224)
    assert out.mode == "RGB"


def test_to_rgb_converts_rgba(make_pil_image):
    from PIL import Image

    rgba = Image.new("RGBA", (10, 10), (1, 2, 3, 128))
    assert to_rgb(rgba).mode == "RGB"


def test_resize_short_side_preserves_aspect(make_pil_image):
    img = make_pil_image(size=(400, 200))
    out = resize_short_side(img, 100)
    assert min(out.size) == 100
    assert out.size == (200, 100)


def test_center_crop_pads_small_image(make_pil_image):
    img = make_pil_image(size=(50, 50))
    out = center_crop(img, 224)
    assert out.size == (224, 224)


def test_auto_orient_idempotent_on_no_exif(make_pil_image):
    img = make_pil_image()
    out = auto_orient(img)
    assert out.size == img.size


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------
def test_factory_selects_siglip2():
    reset_encoder_cache()
    from app.encoders.siglip2 import SigLIP2Encoder

    enc = get_encoder(Settings(encoder="siglip2"))
    assert isinstance(enc, SigLIP2Encoder)
    assert enc.dim == 1152


def test_factory_selects_dinov3():
    reset_encoder_cache()
    from app.encoders.dinov3 import DinoV3Encoder

    enc = get_encoder(Settings(encoder="dinov3"))
    assert isinstance(enc, DinoV3Encoder)
    assert enc.dim == 768


def test_encoders_satisfy_protocol():
    reset_encoder_cache()
    assert isinstance(get_encoder(Settings(encoder="siglip2")), Encoder)


def test_switching_encoder_changes_dim_without_code_change():
    reset_encoder_cache()
    assert get_encoder(Settings(encoder="siglip2")).dim == 1152
    reset_encoder_cache()
    assert get_encoder(Settings(encoder="dinov3")).dim == 768


# --------------------------------------------------------------------------
# Mocked embed path — exercises batching/pooling/normalization without weights
# --------------------------------------------------------------------------
class _FakeTensor:
    """Minimal tensor stand-in supporting the ops used in embed()."""

    def __init__(self, arr: np.ndarray):
        self.arr = np.asarray(arr, dtype=np.float32)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr

    def to(self, device):
        return self

    def half(self):
        return self

    def is_floating_point(self):
        return True

    def mean(self, dim=None):
        return _FakeTensor(self.arr.mean(axis=dim))


class _FakeOutputs:
    def __init__(self, pooled):
        self.pooler_output = pooled
        self.last_hidden_state = None


def _install_fake_torch(monkeypatch):
    fake_torch = types.ModuleType("torch")

    class _InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake_torch.inference_mode = lambda: _InferenceMode()

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    fake_torch.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


def _make_siglip_encoder(monkeypatch, dim, feature_fn):
    _install_fake_torch(monkeypatch)
    reset_encoder_cache()
    enc = get_encoder(Settings(encoder="siglip2", embedding_dim_override=dim, device="cpu", use_fp16=False))

    # Fake processor: returns a dict of fake tensors keyed by the batch.
    enc._processor = mock.Mock(return_value={"pixel_values": _FakeTensor(np.zeros((1, 3, 4, 4)))})

    fake_model = mock.Mock()
    fake_model.get_image_features = feature_fn
    enc._model = fake_model
    enc._torch = sys.modules["torch"]
    enc._device = "cpu"
    return enc


def test_siglip_embed_shape_and_norm(monkeypatch):
    dim = 16

    def feature_fn(**inputs):
        # Deterministic per-call vector.
        return _FakeTensor(np.arange(dim, dtype=np.float32)[None, :] + 1.0)

    enc = _make_siglip_encoder(monkeypatch, dim, feature_fn)
    from PIL import Image

    out = enc.embed([Image.new("RGB", (32, 32), (10, 20, 30))])
    assert out.shape == (1, dim)
    assert out.dtype == np.float32
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)


def test_siglip_embed_unwraps_pooler_output(monkeypatch):
    """Regression: transformers >=5.x returns ``get_image_features`` as a
    ModelOutput wrapper whose image embedding is ``pooler_output`` (not a bare
    tensor). The encoder must unwrap it. Verified live against the real
    google/siglip2-so400m-patch16-naflex model."""
    dim = 16

    def feature_fn(**inputs):
        return _FakeOutputs(_FakeTensor(np.arange(dim, dtype=np.float32)[None, :] + 1.0))

    enc = _make_siglip_encoder(monkeypatch, dim, feature_fn)
    from PIL import Image

    out = enc.embed([Image.new("RGB", (32, 32), (10, 20, 30))])
    assert out.shape == (1, dim)
    assert out.dtype == np.float32
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)


def test_siglip_embed_deterministic_and_identical_cosine(monkeypatch):
    dim = 16

    def feature_fn(**inputs):
        return _FakeTensor(np.linspace(1, 2, dim, dtype=np.float32)[None, :])

    enc = _make_siglip_encoder(monkeypatch, dim, feature_fn)
    from PIL import Image

    img = Image.new("RGB", (32, 32), (10, 20, 30))
    a = enc.embed([img])
    b = enc.embed([img])
    assert np.allclose(a, b)  # deterministic
    cos = float(np.dot(a[0], b[0]))
    assert np.isclose(cos, 1.0, atol=1e-5)  # identical image cosine ≈ 1


def test_siglip_embed_empty_returns_zero_rows(monkeypatch):
    enc = _make_siglip_encoder(monkeypatch, 16, lambda **k: _FakeTensor(np.ones((1, 16))))
    out = enc.embed([])
    assert out.shape == (0, 16)


def test_dinov3_embed_uses_pooler_output(monkeypatch):
    _install_fake_torch(monkeypatch)
    reset_encoder_cache()
    dim = 8
    enc = get_encoder(Settings(encoder="dinov3", embedding_dim_override=dim, device="cpu", use_fp16=False))
    enc._processor = mock.Mock(return_value={"pixel_values": _FakeTensor(np.zeros((1, 3, 4, 4)))})
    fake_model = mock.Mock(return_value=_FakeOutputs(_FakeTensor(np.ones((1, dim), dtype=np.float32))))
    enc._model = fake_model
    enc._torch = sys.modules["torch"]
    enc._device = "cpu"

    from PIL import Image

    out = enc.embed([Image.new("RGB", (32, 32))])
    assert out.shape == (1, dim)
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)


# --------------------------------------------------------------------------
# Guarded real-model test (skips without torch/transformers + weights)
# --------------------------------------------------------------------------
@requires_model
@pytest.mark.slow
def test_real_siglip_shape():  # pragma: no cover - needs model download
    reset_encoder_cache()
    from PIL import Image

    enc = get_encoder(Settings(encoder="siglip2", device="cpu", use_fp16=False))
    out = enc.embed([Image.new("RGB", (224, 224), (10, 20, 30))])
    assert out.shape == (1, enc.dim)
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-3)
