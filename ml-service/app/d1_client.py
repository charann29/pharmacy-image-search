"""D1 REST client (STUB — implemented by Task 2/3).

Used ONLY by the indexing/admin environment (catalog_sync, backfill_index,
reconcile). Reads credentials from settings (D1_ACCOUNT_ID / D1_DATABASE_ID /
D1_API_TOKEN) — literal values only ever come from the environment.

The Cloudflare Worker never uses this; it queries D1 via its binding.
"""
from __future__ import annotations

from typing import Any

from .config import Settings, get_settings


class D1Client:
    """STUB: implemented by Task 2/3."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def query(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError("D1Client.query is implemented by Task 2/3.")
