"""SQLite-backed fake of :class:`app.d1_client.D1Client` for tests.

The indexing scripts issue parameterized SQL with positional ``?`` placeholders
that are identical between Cloudflare D1 and SQLite, so a SQLite database that
has ``0001_init.sql`` applied exercises the *real* script logic end to end
without any network calls. This fake mirrors the ``query`` / ``execute`` /
``batch`` / ``paginate`` surface used by the scripts.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple

Statement = Tuple[str, Optional[Sequence[Any]]]

_MIGRATION = os.path.join(
    os.path.dirname(__file__), "..", "..", "db", "migrations", "0001_init.sql"
)


class FakeD1Client:
    """In-memory SQLite standing in for the D1 REST client."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None) -> None:
        self.conn = conn or sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._apply_migration()

    def _apply_migration(self) -> None:
        with open(os.path.abspath(_MIGRATION), encoding="utf-8") as fh:
            self.conn.executescript(fh.read())
        self.conn.commit()

    def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, list(params) if params else [])
        return [dict(row) for row in cur.fetchall()]

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> dict[str, Any]:
        cur = self.conn.execute(sql, list(params) if params else [])
        self.conn.commit()
        return {"changes": cur.rowcount, "last_row_id": cur.lastrowid}

    def batch(self, statements: Iterable[Statement]) -> list[dict[str, Any]]:
        metas: list[dict[str, Any]] = []
        for sql, params in statements:
            cur = self.conn.execute(sql, list(params) if params else [])
            metas.append({"changes": cur.rowcount, "last_row_id": cur.lastrowid})
        self.conn.commit()
        return metas

    def paginate(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        page_size: int = 500,
    ) -> Iterator[dict[str, Any]]:
        base_params = list(params) if params else []
        offset = 0
        while True:
            rows = self.query(f"{sql} LIMIT ? OFFSET ?", base_params + [page_size, offset])
            if not rows:
                return
            for row in rows:
                yield row
            if len(rows) < page_size:
                return
            offset += page_size

    def close(self) -> None:
        self.conn.close()
