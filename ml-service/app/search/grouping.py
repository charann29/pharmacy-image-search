"""Group image hits by product_id and pool to a ranked product list (Task 6).

ANN returns per-image hits; a product may have several images in the top-K.
:func:`group_by_product` pools those into one score per product (default
``max``; ``mean`` configurable) and returns a list ranked by pooled score.
"""
from __future__ import annotations

from typing import Any, Iterable

VALID_POOLS = ("max", "mean")


def _score(hit: dict[str, Any]) -> float:
    return float(hit.get("score", 0.0))


def _product_id(hit: dict[str, Any]) -> Any:
    # product_id may live at the top level or under a Qdrant payload.
    if "product_id" in hit:
        return hit["product_id"]
    payload = hit.get("payload") or {}
    return payload.get("product_id")


def group_by_product(
    hits: Iterable[dict[str, Any]],
    pool: str = "max",
) -> list[dict[str, Any]]:
    """Group image hits by product_id and pool scores.

    Args:
        hits: iterable of ``{score, payload:{product_id,...}}`` (or top-level
            ``product_id``) ANN hits.
        pool: ``"max"`` (default) or ``"mean"``.

    Returns:
        ``[{product_id, score, image_count}]`` sorted by pooled score desc.
        Ties break deterministically by product_id.
    """
    if pool not in VALID_POOLS:
        raise ValueError(f"pool must be one of {VALID_POOLS}, got {pool!r}")

    buckets: dict[Any, list[float]] = {}
    for hit in hits:
        pid = _product_id(hit)
        if pid is None:
            continue
        buckets.setdefault(pid, []).append(_score(hit))

    products: list[dict[str, Any]] = []
    for pid, scores in buckets.items():
        pooled = max(scores) if pool == "max" else sum(scores) / len(scores)
        products.append(
            {"product_id": pid, "score": float(pooled), "image_count": len(scores)}
        )

    products.sort(key=lambda p: (-p["score"], str(p["product_id"])))
    return products
