"""Tests for qdrant_client: collection config, deterministic ids, upsert,
query, and startup validation.

Uses in-memory Qdrant (``QdrantClient(":memory:")``) when qdrant-client is
installed — this exercises the real model objects (VectorParams, HnswConfigDiff,
ScalarQuantization, SearchParams) against the pinned client version, satisfying
the "model objects accepted by qdrant-client 1.12.1" requirement without a live
server. Tests skip when qdrant-client is absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.search import qdrant_client as qc

from conftest import requires_qdrant


# --------------------------------------------------------------------------
# Deterministic ids (no qdrant dep)
# --------------------------------------------------------------------------
def test_deterministic_vector_id_stable():
    a = qc.deterministic_vector_id("siglip2", 1152, "img-1")
    b = qc.deterministic_vector_id("siglip2", 1152, "img-1")
    assert a == b


def test_deterministic_vector_id_varies_by_encoder_dim_image():
    base = qc.deterministic_vector_id("siglip2", 1152, "img-1")
    assert base != qc.deterministic_vector_id("dinov3", 768, "img-1")
    assert base != qc.deterministic_vector_id("siglip2", 768, "img-1")
    assert base != qc.deterministic_vector_id("siglip2", 1152, "img-2")


# --------------------------------------------------------------------------
# collection-name encoder parsing
# --------------------------------------------------------------------------
def test_encoder_from_collection_name():
    tmpl = "image_embeddings_{encoder}_{dim}"
    assert qc.encoder_from_collection_name("image_embeddings_siglip2_1152", tmpl) == "siglip2"
    assert qc.encoder_from_collection_name("image_embeddings_dinov3_768", tmpl) == "dinov3"
    assert qc.encoder_from_collection_name("totally_other", tmpl) is None


# --------------------------------------------------------------------------
# Model-object builders accepted by the pinned client version
# --------------------------------------------------------------------------
@requires_qdrant
def test_model_builders_construct():
    vp = qc.build_vectors_config(1152)
    assert vp.size == 1152
    hnsw = qc.build_hnsw_config()
    assert hnsw.m == 16 and hnsw.ef_construct == 128
    quant = qc.build_quantization_config()
    assert quant.scalar.always_ram is True
    sp = qc.build_rescore_search_params()
    assert sp.quantization.rescore is True


# --------------------------------------------------------------------------
# In-memory Qdrant integration
# --------------------------------------------------------------------------
@requires_qdrant
def test_ensure_collection_two_different_dims():
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    settings = Settings(encoder="siglip2")

    n1 = qc.ensure_collection(
        settings, client=client, collection_name="image_embeddings_siglip2_1152", embedding_dim=1152
    )
    n2 = qc.ensure_collection(
        settings, client=client, collection_name="image_embeddings_dinov3_768", embedding_dim=768
    )
    assert client.collection_exists(n1)
    assert client.collection_exists(n2)
    assert client.get_collection(n1).config.params.vectors.size == 1152
    assert client.get_collection(n2).config.params.vectors.size == 768


@requires_qdrant
def test_ensure_collection_idempotent():
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    settings = Settings(encoder="siglip2")
    name = "image_embeddings_siglip2_1152"
    qc.ensure_collection(settings, client=client, collection_name=name, embedding_dim=1152)
    # Second call must not raise (collection already exists).
    qc.ensure_collection(settings, client=client, collection_name=name, embedding_dim=1152)
    assert client.collection_exists(name)


@requires_qdrant
def test_deterministic_upsert_idempotent_and_query_returns_point():
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    settings = Settings(encoder="siglip2", embedding_dim_override=8)
    name = qc.ensure_collection(settings, client=client)

    vec = np.zeros(8, dtype=np.float32)
    vec[0] = 1.0
    point = qc.build_point(
        vec, image_id="img-1", product_id="prod-1", encoder="siglip2", embedding_dim=8, is_reference=True
    )
    qc.upsert([point], settings=settings, client=client, collection_name=name)
    qc.upsert([point], settings=settings, client=client, collection_name=name)  # same id
    assert client.count(name).count == 1  # idempotent

    hits = qc.query(vec, top_k=5, settings=settings, client=client, collection_name=name)
    assert len(hits) == 1
    assert hits[0]["payload"]["product_id"] == "prod-1"
    assert hits[0]["payload"]["is_reference"] is True


# --------------------------------------------------------------------------
# Startup validation
# --------------------------------------------------------------------------
@requires_qdrant
def test_startup_validation_passes_when_consistent():
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    settings = Settings(encoder="siglip2")
    qc.ensure_collection(settings, client=client)
    # Should not raise.
    qc.validate_active_collection(settings, client=client)


@requires_qdrant
def test_startup_validation_raises_on_missing_collection():
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    settings = Settings(encoder="siglip2")
    with pytest.raises(qc.StartupValidationError):
        qc.validate_active_collection(settings, client=client)


@requires_qdrant
def test_startup_validation_raises_on_dim_mismatch():
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    # Create a collection at the active name but with the WRONG dim.
    settings = Settings(encoder="siglip2")
    name = settings.active_collection
    qc.ensure_collection(settings, client=client, collection_name=name, embedding_dim=768)
    with pytest.raises(qc.StartupValidationError):
        qc.validate_active_collection(settings, client=client)


@requires_qdrant
def test_startup_validation_raises_on_encoder_mismatch():
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    # ENCODER=siglip2 (dim 1152) but ACTIVE_COLLECTION points at a dinov3 name.
    settings = Settings(encoder="siglip2", active_collection_override="image_embeddings_dinov3_1152")
    qc.ensure_collection(settings, client=client, embedding_dim=1152)
    with pytest.raises(qc.StartupValidationError):
        qc.validate_active_collection(settings, client=client)
