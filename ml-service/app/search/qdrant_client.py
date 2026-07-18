"""Qdrant collection config, upsert, and ANN query (STUB — implemented by Task 2/6).

Per-encoder collections: ``image_embeddings_{encoder}_{dim}``. HNSW + int8
scalar quantization with rescore. Deterministic vector ids via uuid5.
"""
from __future__ import annotations

import uuid
from typing import Any

from ..config import Settings, get_settings

# Stable namespace for deterministic vector ids.
VECTOR_ID_NAMESPACE = uuid.UUID("6f4d2e3a-1c8b-4b7e-9a2d-000000000001")


def deterministic_vector_id(encoder: str, embedding_dim: int, image_id: str) -> str:
    """uuid5(namespace, f"{encoder}:{embedding_dim}:{image_id}") — idempotent upserts."""
    return str(uuid.uuid5(VECTOR_ID_NAMESPACE, f"{encoder}:{embedding_dim}:{image_id}"))


def ensure_collection(settings: Settings | None = None) -> None:
    """Create the active collection (VectorParams + HNSW + quantization).

    STUB: implemented by Task 2.
    """
    raise NotImplementedError("ensure_collection is implemented by Task 2.")


def upsert(points: list[dict[str, Any]], settings: Settings | None = None) -> None:
    """STUB: implemented by Task 2/11."""
    raise NotImplementedError("upsert is implemented by Task 2/11.")


def query(vector: Any, top_k: int = 200, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Single dense ANN on ACTIVE_COLLECTION with rescore. STUB: Task 6."""
    raise NotImplementedError("query is implemented by Task 6.")
