from __future__ import annotations

from typing import Any


class IsofoodCenterError(Exception):
    """Base class for every error raised by this library."""


class AuthenticationError(IsofoodCenterError):
    """Login failed, or a request was made without a valid session token."""


class APIError(IsofoodCenterError):
    """The server returned an error response.

    Attributes:
        status_code: The HTTP status code, if known.
        payload: The parsed JSON error body from the server, if any.
    """

    def __init__(self, message: str, status_code: int | None = None, payload: Any | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class NotFoundError(APIError):
    """The requested record does not exist (HTTP 404)."""


class ValidationError(APIError):
    """The server rejected the payload (HTTP 422). `.payload` holds the field errors."""
