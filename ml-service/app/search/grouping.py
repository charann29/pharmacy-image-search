"""Group image hits by product_id (STUB — implemented by Task 6).

Pools per-product scores (default max, mean configurable) into a ranked list.
"""
from __future__ import annotations

from typing import Any


def group_by_product(hits: list[dict[str, Any]], pool: str = "max") -> list[dict[str, Any]]:
    """Return ranked [{product_id, score}] grouped from image hits. STUB: Task 6."""
    raise NotImplementedError("group_by_product is implemented by Task 6.")
