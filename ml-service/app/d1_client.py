"""Cloudflare D1 REST client (indexing/admin environment ONLY).

Used by the indexing scripts (``catalog_sync``, ``backfill_index``,
``reconcile``). The Cloudflare Worker never touches this — it queries D1 via its
binding. Credentials are read from :class:`~app.config.Settings`
(``D1_ACCOUNT_ID`` / ``D1_DATABASE_ID`` / ``D1_API_TOKEN``); literal values only
ever arrive from the environment, never from source.

The REST surface used here is::

    POST /accounts/{account_id}/d1/database/{database_id}/query
        body: {"sql": "...", "params": ["...", ...]}

All SQL is parameterized with positional ``?`` placeholders — identical between
D1 and SQLite, which lets a SQLite-backed fake exercise the same script logic in
tests. Read/write/batch helpers and a pagination helper are provided.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple

from .config import Settings, get_settings

# A single batch statement: (sql, params).
Statement = Tuple[str, Optional[Sequence[Any]]]

CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"


class D1ConfigError(RuntimeError):
    """Raised when required D1 REST credentials are missing from the environment."""


class D1RequestError(RuntimeError):
    """Raised when the D1 REST API returns a non-success response."""


class D1Client:
    """Thin D1 REST client with parameterized read/write/batch + pagination.

    Parameters
    ----------
    settings:
        Optional settings override; defaults to the cached process settings.
    session:
        Optional pre-built ``httpx.Client`` (injected in tests). When omitted a
        client is created lazily on first request using the configured token.
    timeout:
        Per-request timeout (seconds).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        session: Any = None,
        timeout: float = 30.0,
    ) -> None:
        self._settings = settings or get_settings()
        self._session = session
        self._timeout = timeout
        self._owns_session = session is None

    # -- configuration ---------------------------------------------------
    def _require_creds(self) -> tuple[str, str, str]:
        s = self._settings
        missing = [
            name
            for name, val in (
                ("D1_ACCOUNT_ID", s.d1_account_id),
                ("D1_DATABASE_ID", s.d1_database_id),
                ("D1_API_TOKEN", s.d1_api_token),
            )
            if not val
        ]
        if missing:
            raise D1ConfigError(
                "Missing D1 REST credentials (set via environment): "
                + ", ".join(missing)
            )
        return s.d1_account_id, s.d1_database_id, s.d1_api_token  # type: ignore[return-value]

    def _endpoint(self, kind: str = "query") -> str:
        account_id, database_id, _ = self._require_creds()
        return (
            f"{CLOUDFLARE_API_BASE}/accounts/{account_id}"
            f"/d1/database/{database_id}/{kind}"
        )

    def _client(self):
        if self._session is not None:
            return self._session
        import httpx  # noqa: WPS433 (local import keeps module importable w/o httpx)

        _, _, token = self._require_creds()
        self._session = httpx.Client(
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        return self._session

    # -- low-level request ----------------------------------------------
    def _post(self, sql: str, params: Optional[Sequence[Any]]) -> list[dict[str, Any]]:
        """POST a single statement, returning the D1 ``result`` array.

        Each element carries ``results`` (row dicts), ``success`` and ``meta``.
        """
        client = self._client()
        payload = {"sql": sql, "params": list(params) if params else []}
        resp = client.post(self._endpoint("query"), content=json.dumps(payload))
        # httpx.Response — raise for HTTP errors, then check the CF envelope.
        if resp.status_code >= 400:
            raise D1RequestError(
                f"D1 HTTP {resp.status_code}: {resp.text[:500]}"
            )
        body = resp.json()
        if not body.get("success", False):
            raise D1RequestError(f"D1 API error: {body.get('errors')}")
        return body.get("result", [])

    # -- public helpers --------------------------------------------------
    def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> list[dict[str, Any]]:
        """Run a read statement and return its rows as a list of dicts."""
        result = self._post(sql, params)
        if not result:
            return []
        # A single statement -> one result element carrying `results`.
        return list(result[0].get("results", []) or [])

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> dict[str, Any]:
        """Run a write statement and return its ``meta`` (changes, last_row_id...)."""
        result = self._post(sql, params)
        if not result:
            return {}
        return dict(result[0].get("meta", {}) or {})

    def batch(self, statements: Iterable[Statement]) -> list[dict[str, Any]]:
        """Execute a sequence of (sql, params) statements.

        Statements run sequentially; each returns its ``meta``. Sequential
        execution keeps behaviour identical between the REST client and the
        SQLite-backed fake used in tests.
        """
        metas: list[dict[str, Any]] = []
        for sql, params in statements:
            metas.append(self.execute(sql, params))
        return metas

    def paginate(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        page_size: int = 500,
    ) -> Iterator[dict[str, Any]]:
        """Yield rows for ``sql`` in pages using appended ``LIMIT ? OFFSET ?``.

        ``sql`` must NOT already contain a LIMIT/OFFSET clause; a stable
        ``ORDER BY`` is the caller's responsibility for deterministic paging.
        """
        base_params = list(params) if params else []
        offset = 0
        while True:
            page_sql = f"{sql} LIMIT ? OFFSET ?"
            rows = self.query(page_sql, base_params + [page_size, offset])
            if not rows:
                return
            for row in rows:
                yield row
            if len(rows) < page_size:
                return
            offset += page_size

    def close(self) -> None:
        """Close the underlying httpx client if this instance created it."""
        if self._owns_session and self._session is not None:
            close = getattr(self._session, "close", None)
            if callable(close):
                close()
            self._session = None
