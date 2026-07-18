"""Central configuration for the image-search ML service.

All values are read from environment variables (or an optional ``.env`` file).
Secrets are referenced by NAME only — never hard-code literal secret values here.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

EncoderName = Literal["dinov3", "siglip2"]

# Default embedding dimensions per encoder (used when EMBEDDING_DIM is unset).
# DINOv3 ViT-B/16 -> 768; SigLIP2 (so400m/large) image tower -> 1152.
_DEFAULT_DIMS: dict[str, int] = {
    "dinov3": 768,
    "siglip2": 1152,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # ---- Encoder selection (encoder-agnostic design) --------------------
    encoder: EncoderName = Field(
        default="siglip2",
        description="Active image encoder. SigLIP2 is the Apache-2.0 launch encoder; "
        "DINOv3 is gated behind Task 0 license/access approval.",
    )
    # If unset, resolved from the encoder default via `embedding_dim` property.
    embedding_dim_override: Optional[int] = Field(
        default=None,
        alias="EMBEDDING_DIM",
        description="Explicit embedding dimension. Defaults to the encoder's native dim.",
    )

    # ---- Qdrant ----------------------------------------------------------
    qdrant_url: str = Field(
        default="http://qdrant:6333",
        description="Base URL for the Qdrant service.",
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Optional Qdrant API key (secret name only).",
    )
    # Collection naming template. Rendered as image_embeddings_{encoder}_{dim}.
    collection_name_template: str = Field(
        default="image_embeddings_{encoder}_{dim}",
        description="Template for per-encoder collection names.",
    )
    # If unset, resolved from encoder + dim via `active_collection` property.
    active_collection_override: Optional[str] = Field(
        default=None,
        alias="ACTIVE_COLLECTION",
        description="Collection the query path targets. Defaults to the rendered "
        "template for the active encoder/dim. Flip this to swap encoders.",
    )

    # ---- Auth (Worker -> service) ---------------------------------------
    ml_service_shared_secret: Optional[str] = Field(
        default=None,
        description="Shared secret / HMAC signing key for Worker->service auth. "
        "Secret name only — set the real value via the environment.",
    )
    auth_replay_window_seconds: int = Field(
        default=300,
        description="Max allowed clock skew / replay window for HMAC signatures.",
    )
    auth_require: bool = Field(
        default=True,
        description="When False, auth is bypassed (local dev only).",
    )

    # ---- Startup validation ---------------------------------------------
    validate_collection_on_startup: bool = Field(
        default=True,
        description="Validate active collection encoder/dim against config on boot. "
        "A genuine mismatch blocks startup; disable only for tests/dev.",
    )

    # ---- Weak-match threshold calibration (Task 10) ---------------------
    weak_match_threshold_file: Optional[str] = Field(
        default=None,
        alias="WEAK_MATCH_THRESHOLD_FILE",
        description="Path to the calibrated per-encoder weak-match threshold JSON "
        "written by eval/run_eval.py. When set and the active encoder is present, "
        "its calibrated value overrides the provisional default (0.35).",
    )

    # ---- Inference -------------------------------------------------------
    batch_size: int = Field(default=32, description="GPU embedding batch size.")
    device: str = Field(default="cuda", description="Torch device: cuda | cpu.")
    use_fp16: bool = Field(default=True, description="Use fp16 on GPU.")

    # ---- Search ----------------------------------------------------------
    search_top_k: int = Field(default=200, description="ANN top-K for /search.")
    search_pool: str = Field(
        default="max",
        description="Per-product score pooling for /search: 'max' or 'mean'.",
    )

    # ---- D1 REST (indexing environment ONLY) ----------------------------
    # These are used by scripts/ (catalog_sync, backfill_index) in the admin/
    # indexing environment. The Worker uses a D1 *binding* and never these.
    d1_account_id: Optional[str] = Field(default=None, description="Cloudflare account id (secret name only).")
    d1_database_id: Optional[str] = Field(default=None, description="D1 database id (secret name only).")
    d1_api_token: Optional[str] = Field(default=None, description="Cloudflare API token (secret name only).")

    # ---- Service ---------------------------------------------------------
    log_level: str = Field(default="INFO")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, description="Max request image size.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embedding_dim(self) -> int:
        if self.embedding_dim_override is not None:
            return self.embedding_dim_override
        return _DEFAULT_DIMS[self.encoder]

    def render_collection_name(self, encoder: str, dim: int) -> str:
        return self.collection_name_template.format(encoder=encoder, dim=dim)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def active_collection(self) -> str:
        if self.active_collection_override:
            return self.active_collection_override
        return self.render_collection_name(self.encoder, self.embedding_dim)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
