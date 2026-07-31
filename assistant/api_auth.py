"""Bearer-token gate for the cost-incurring HTTP routes.

Single shared token, no user model — enough to keep a public demo URL from
running up an Anthropic bill, and deliberately not more than that.

Enforcement is conditional: with `ASSISTANT_API_TOKEN` unset the dependency is
a no-op, so local dev and the test suite run unchanged. Setting the env var is
itself the deploy step; there is no separate flag to remember.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status

from assistant import config


def require_auth(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Reject the request unless it carries the configured bearer token.

    No token configured means auth is disabled and every request passes.
    """
    token = config.get_api_token()
    if token is None:
        return

    expected = f"Bearer {token}"

    # Constant-time compare: a plain `!=` short-circuits at the first differing
    # byte, which leaks the token's prefix through response timing.
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
