"""Worker -> ML-service authentication.

Two supported mechanisms (either passes):

1. **HMAC-signed request** (preferred, replay-resistant):
   - Header ``X-Signature``  = hex(HMAC_SHA256(key, f"{timestamp}.{method}.{path}"))
   - Header ``X-Timestamp``  = unix seconds (int)
   The request is rejected if the timestamp is outside the configured replay
   window (default 300s) or the signature does not match.

2. **Shared-secret bearer** (fallback):
   - Header ``Authorization: Bearer <ML_SERVICE_SHARED_SECRET>``

Both compare using ``hmac.compare_digest`` (constant-time). Failure raises
``HTTPException(401)``. The signing/secret key is read from settings
(``ML_SERVICE_SHARED_SECRET``) — the literal value only ever lives in the
environment, never in source.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings


def compute_signature(key: str, timestamp: str, method: str, path: str) -> str:
    """Deterministic HMAC-SHA256 signature used by both signer and verifier."""
    message = f"{timestamp}.{method.upper()}.{path}".encode("utf-8")
    return hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest()


def _verify_hmac(
    key: str,
    signature: str,
    timestamp: str,
    method: str,
    path: str,
    replay_window_seconds: int,
    now: Optional[float] = None,
) -> bool:
    # Timestamp must be a valid int within the replay window.
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    current = int(now if now is not None else time.time())
    if abs(current - ts) > replay_window_seconds:
        return False
    expected = compute_signature(key, timestamp, method, path)
    return hmac.compare_digest(expected, signature or "")


def _verify_bearer(key: str, authorization: Optional[str]) -> bool:
    if not authorization:
        return False
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return False
    token = authorization[len(prefix):]
    return hmac.compare_digest(token, key)


def verify_request(request: Request, settings: Settings) -> bool:
    """Return True if the request carries valid auth, else False.

    Pure-ish helper (only reads request headers) so it is unit-testable
    without a running server.
    """
    if not settings.auth_require:
        return True

    key = settings.ml_service_shared_secret
    if not key:
        # Fail closed: auth is required but no key configured.
        return False

    signature = request.headers.get("x-signature")
    timestamp = request.headers.get("x-timestamp")
    if signature and timestamp:
        return _verify_hmac(
            key=key,
            signature=signature,
            timestamp=timestamp,
            method=request.method,
            path=request.url.path,
            replay_window_seconds=settings.auth_replay_window_seconds,
        )

    return _verify_bearer(key, request.headers.get("authorization"))


async def require_auth(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI dependency: raises 401 on missing/invalid auth."""
    if not verify_request(request, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication.",
            headers={"WWW-Authenticate": "Bearer"},
        )
