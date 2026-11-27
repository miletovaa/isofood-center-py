from __future__ import annotations

from typing import Any

from .client import OrgClient
from .config import resolve_credentials
from .exceptions import (
    APIError,
    AuthenticationError,
    NotFoundError,
    IsofoodCenterError,
    ValidationError,
)


def connect(
    base_url: str | None = None,
    email: str | None = None,
    password: str | None = None,
    token: str | None = None,
    profile: str = "default",
    **kwargs: Any,
) -> OrgClient:
    """The main entry point: connect(base_url, email, password) -> OrgClient.

    Any argument left out is resolved from environment variables
    (ISOFOOD_CENTER_BASE_URL / ISOFOOD_CENTER_EMAIL / ISOFOOD_CENTER_PASSWORD) or ~/.isofood_center/config.ini
    (see config.py). Pass `token` instead of email/password to skip login and
    reuse an existing bearer token.

    Args:
        base_url: Root URL of the org_dma instance.
        email: Login email.
        password: Login password.
        token: An existing bearer token to reuse instead of logging in.
        profile: Config-file section to fall back to for any value not given
            explicitly and not found in an environment variable.
        **kwargs: Passed through to `OrgClient` (e.g. `timeout`, `verify`).

    Returns:
        A connected `OrgClient`.
    """
    base_url, email, password = resolve_credentials(base_url, email, password, profile=profile)
    return OrgClient(base_url=base_url, email=email, password=password, token=token, **kwargs)


__all__ = [
    "connect",
    "OrgClient",
    "IsofoodCenterError",
    "AuthenticationError",
    "APIError",
    "NotFoundError",
    "ValidationError",
]
