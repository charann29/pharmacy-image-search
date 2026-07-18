"""Encoder factory — select the active encoder by ``ENCODER`` setting.

``get_encoder(settings)`` returns a lazily-loaded encoder instance implementing
the :class:`~app.encoders.base.Encoder` protocol. The instance is cached per
process so the model is loaded once.
"""
from __future__ import annotations

from typing import Optional

from ..config import Settings, get_settings
from .base import Encoder, l2_normalize

__all__ = ["Encoder", "get_encoder", "l2_normalize", "reset_encoder_cache"]

_ENCODER_CACHE: dict[str, Encoder] = {}


def _build_encoder(settings: Settings) -> Encoder:
    if settings.encoder == "dinov3":
        from .dinov3 import DinoV3Encoder

        return DinoV3Encoder(settings)
    if settings.encoder == "siglip2":
        from .siglip2 import SigLIP2Encoder

        return SigLIP2Encoder(settings)
    raise ValueError(f"Unknown encoder: {settings.encoder!r}")


def get_encoder(settings: Optional[Settings] = None) -> Encoder:
    """Return the active encoder instance (cached per encoder name)."""
    settings = settings or get_settings()
    key = f"{settings.encoder}:{settings.embedding_dim}"
    cached = _ENCODER_CACHE.get(key)
    if cached is None:
        cached = _build_encoder(settings)
        _ENCODER_CACHE[key] = cached
    return cached


def reset_encoder_cache() -> None:
    """Clear the encoder cache (used in tests / after config changes)."""
    _ENCODER_CACHE.clear()
