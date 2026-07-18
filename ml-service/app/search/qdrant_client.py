"""Qdrant collection config, upsert, ANN query, and startup validation.

Per-encoder collections named ``image_embeddings_{encoder}_{dim}`` (e.g.
``image_embeddings_siglip2_1152``). Each encoder writes its own collection so
several can coexist; ``ACTIVE_COLLECTION`` selects which the query path uses.

Key behaviours:
  * ``ensure_collection()`` creates the collection with cosine distance, HNSW
    (m=16, ef_construct=128) and int8 scalar quantization (always_ram).
  * Vector ids are deterministic (uuid5) so re-runs upsert instead of
    duplicating, keeping Qdrant<->D1 writes idempotent.
  * ``query()`` runs a single dense ANN with quantization rescore for accuracy.
  * ``validate_active_collection()`` blocks startup on any encoder/dim/name
    mismatch between config and the live collection.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional, Sequence

from ..config import Settings, get_settings

# Stable namespace for deterministic vector ids.
VECTOR_ID_NAMESPACE = uuid.UUID("6f4d2e3a-1c8b-4b7e-9a2d-000000000001")

# Encoder metadata is stored on each point payload; the collection's encoder is
# derived from its name (image_embeddings_{encoder}_{dim}) at validation time.
DEFAULT_TOP_K = 200


def deterministic_vector_id(encoder: str, embedding_dim: int, image_id: str) -> str:
    """uuid5(namespace, f"{encoder}:{embedding_dim}:{image_id}") — idempotent upserts."""
    return str(uuid.uuid5(VECTOR_ID_NAMESPACE, f"{encoder}:{embedding_dim}:{image_id}"))


def _import_models():
    """Import qdrant_client.models lazily so the module imports without the dep."""
    from qdrant_client import models  # noqa: WPS433 (local import by design)

    return models


def get_client(settings: Optional[Settings] = None):
    """Construct a QdrantClient from settings.

    Lazy import keeps the module importable in environments without
    qdrant-client (e.g. partial test installs).
    """
    settings = settings or get_settings()
    from qdrant_client import QdrantClient  # noqa: WPS433

    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def build_vectors_config(embedding_dim: int):
    """VectorParams(size=dim, distance=COSINE)."""
    models = _import_models()
    return models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)


def build_hnsw_config():
    """HnswConfigDiff(m=16, ef_construct=128)."""
    models = _import_models()
    return models.HnswConfigDiff(m=16, ef_construct=128)


def build_quantization_config():
    """ScalarQuantization(INT8, always_ram=True)."""
    models = _import_models()
    return models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            always_ram=True,
        )
    )


def build_rescore_search_params():
    """SearchParams(quantization=QuantizationSearchParams(rescore=True))."""
    models = _import_models()
    return models.SearchParams(
        quantization=models.QuantizationSearchParams(rescore=True)
    )


def ensure_collection(
    settings: Optional[Settings] = None,
    client: Any = None,
    collection_name: Optional[str] = None,
    embedding_dim: Optional[int] = None,
) -> str:
    """Create the collection (if missing) with the pinned config.

    Returns the collection name. Idempotent: skips creation if it already
    exists. ``collection_name``/``embedding_dim`` override the active defaults
    (used to build a second, differently-dimensioned collection in tests and
    during encoder swaps).
    """
    settings = settings or get_settings()
    client = client or get_client(settings)
    collection_name = collection_name or settings.active_collection
    embedding_dim = embedding_dim if embedding_dim is not None else settings.embedding_dim

    if client.collection_exists(collection_name):
        return collection_name

    client.create_collection(
        collection_name=collection_name,
        vectors_config=build_vectors_config(embedding_dim),
        hnsw_config=build_hnsw_config(),
        quantization_config=build_quantization_config(),
    )
    return collection_name


def build_point(
    vector: Sequence[float],
    *,
    image_id: str,
    product_id: str,
    encoder: str,
    embedding_dim: int,
    is_reference: bool = False,
    vector_id: Optional[str] = None,
):
    """Build a PointStruct with a deterministic id and the standard payload."""
    models = _import_models()
    vid = vector_id or deterministic_vector_id(encoder, embedding_dim, image_id)
    return models.PointStruct(
        id=vid,
        vector=list(vector),
        payload={
            "product_id": product_id,
            "image_id": image_id,
            "encoder": encoder,
            "is_reference": bool(is_reference),
        },
    )


def upsert(
    points: list,
    settings: Optional[Settings] = None,
    client: Any = None,
    collection_name: Optional[str] = None,
) -> None:
    """Upsert PointStruct objects into the (active) collection.

    Deterministic ids make this idempotent: re-upserting the same image
    overwrites rather than duplicating.
    """
    settings = settings or get_settings()
    client = client or get_client(settings)
    collection_name = collection_name or settings.active_collection
    client.upsert(collection_name=collection_name, points=points)


def query(
    vector: Any,
    top_k: int = DEFAULT_TOP_K,
    settings: Optional[Settings] = None,
    client: Any = None,
    collection_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Single dense ANN on the active collection with quantization rescore.

    Returns a list of ``{id, score, payload}`` dicts (payload carries
    product_id/image_id/encoder/is_reference).
    """
    settings = settings or get_settings()
    client = client or get_client(settings)
    collection_name = collection_name or settings.active_collection

    vector_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    result = client.query_points(
        collection_name=collection_name,
        query=vector_list,
        limit=top_k,
        with_payload=True,
        search_params=build_rescore_search_params(),
    )
    return [
        {"id": p.id, "score": p.score, "payload": p.payload or {}}
        for p in result.points
    ]


def encoder_from_collection_name(name: str, template: str) -> Optional[str]:
    """Best-effort parse of the encoder token from a collection name.

    Template is ``image_embeddings_{encoder}_{dim}``. Returns None if the name
    does not match the template's fixed prefix/suffix shape.
    """
    # Split the template into literal parts around the {encoder} and {dim}.
    prefix, _, rest = template.partition("{encoder}")
    mid, _, suffix = rest.partition("{dim}")  # mid is the literal between tokens
    if not name.startswith(prefix):
        return None
    body = name[len(prefix):]
    # body == f"{encoder}{mid}{dim}{suffix}"; strip suffix then split on mid.
    if suffix and body.endswith(suffix):
        body = body[: len(body) - len(suffix)]
    if mid and mid in body:
        encoder_part, _, _dim_part = body.rpartition(mid)
        return encoder_part or None
    return None


class StartupValidationError(RuntimeError):
    """Raised when the active collection is inconsistent with the loaded encoder."""


def validate_active_collection(
    settings: Optional[Settings] = None,
    client: Any = None,
) -> None:
    """Block startup unless the active collection matches the loaded encoder.

    Asserts:
      1. the active collection exists;
      2. its vector dim == the configured EMBEDDING_DIM;
      3. its encoder (parsed from the collection name) == ENCODER.

    Raises ``StartupValidationError`` on any mismatch so the service refuses to
    serve wrong/failed queries (e.g. ENCODER=siglip2 pointed at a dinov3
    collection).
    """
    settings = settings or get_settings()
    client = client or get_client(settings)
    name = settings.active_collection

    if not client.collection_exists(name):
        raise StartupValidationError(
            f"Active collection {name!r} does not exist. Index it before startup "
            f"(encoder={settings.encoder}, dim={settings.embedding_dim})."
        )

    info = client.get_collection(name)
    # vectors config may be a single VectorParams or a mapping (named vectors).
    vectors = info.config.params.vectors
    actual_dim = getattr(vectors, "size", None)
    if actual_dim is None and isinstance(vectors, dict):
        # Unnamed default vector or first named vector.
        first = next(iter(vectors.values()), None)
        actual_dim = getattr(first, "size", None)

    if actual_dim != settings.embedding_dim:
        raise StartupValidationError(
            f"Active collection {name!r} dim {actual_dim} != loaded encoder dim "
            f"{settings.embedding_dim} (encoder={settings.encoder})."
        )

    parsed_encoder = encoder_from_collection_name(name, settings.collection_name_template)
    if parsed_encoder is not None and parsed_encoder != settings.encoder:
        raise StartupValidationError(
            f"Active collection {name!r} encoder {parsed_encoder!r} != loaded "
            f"ENCODER {settings.encoder!r}."
        )
