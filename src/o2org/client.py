"""HTTP client for an o2_org_dma instance: login/logout and generic verbs.

Talks to a REST API that (per the o2_org_dma README) is expected to expose:

    POST /api/login   {email, password} -> {token, user}
    POST /api/logout
    GET  /api/me       -> user

and standard Laravel apiResource routes (index/store/show/update/destroy) for
everything else. See resources.py for the per-entity wrappers, and the README
for the full expected contract.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Any

import requests

from .exceptions import APIError, AuthenticationError, NotFoundError, ValidationError

if TYPE_CHECKING:
    from .resources import ExperimentRecords, ProjectCompounds, Resource, Samples


def _safe_json(response: requests.Response) -> Any | None:
    try:
        return response.json()
    except ValueError:
        return None


class OrgClient:
    """A connection to one o2_org_dma instance.

    Normally created via `o2org.connect(...)` rather than directly. Every
    resource below is attached as an attribute once construction finishes (by
    `resources.attach_resources()`) — the declarations here exist only so your
    editor can autocomplete and hover them; the real values are set at runtime.
    """

    # -- retrieval/CRUD wrappers, one per entity (see resources.py) ----------
    compounds: Resource
    sources: Resource
    projects: Resource
    compound_synonyms: Resource
    retention_indices: Resource
    diseases: Resource
    compound_disease_associations: Resource
    ontologies: Resource
    compound_ontologies: Resource
    biomarkers: Resource
    compound_biomarkers: Resource
    taxonomies: Resource
    samplings: Resource
    experiments: Resource
    project_compounds: ProjectCompounds
    experiment_records: ExperimentRecords
    samples: Samples

    def __init__(
        self,
        base_url: str,
        email: str | None = None,
        password: str | None = None,
        token: str | None = None,
        timeout: int = 30,
        verify: bool = True,
    ) -> None:
        """Args:
            base_url: Root URL of the org_dma instance, e.g. "https://dma.example.org".
                A trailing slash is stripped if present.
            email: Login email. Ignored if `token` is given.
            password: Login password. Ignored if `token` is given.
            token: An existing bearer token to reuse instead of logging in.
            timeout: Request timeout in seconds, applied to every HTTP call.
            verify: Whether to verify TLS certificates (passed to `requests`).
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = verify
        self.session.headers.update({"Accept": "application/json"})

        self._email = email
        self._token: str | None = None
        self._user: dict[str, Any] | None = None

        if token:
            self._set_token(token)
        elif email and password:
            self.login(email, password)

        from .resources import attach_resources
        attach_resources(self)

    # -- authentication -----------------------------------------------------

    def login(self, email: str, password: str) -> dict[str, Any] | None:
        """Exchange email/password for a bearer token via POST /api/login.

        Args:
            email: Login email.
            password: Login password.

        Returns:
            The `user` object from the login response, if the server included one.

        Raises:
            AuthenticationError: Credentials were rejected, or the response had
                no `token`/`access_token` field.
        """
        data = self._request("POST", "/api/login",
                              json={"email": email, "password": password},
                              auth_required=False)
        token = (data or {}).get("token") or (data or {}).get("access_token")
        if not token:
            raise AuthenticationError(
                "Login succeeded but the response had no 'token' or "
                "'access_token' field. Check the /api/login contract."
            )
        self._email = email
        self._user = (data or {}).get("user")
        self._set_token(token)
        return self._user

    def logout(self) -> None:
        """Invalidate the current token via POST /api/logout.

        Does nothing if no token is set (e.g. logout() called twice).
        """
        if not self._token:
            return
        try:
            self._request("POST", "/api/logout")
        finally:
            self._token = None
            self.session.headers.pop("Authorization", None)

    def me(self) -> dict[str, Any] | None:
        """Fetch the current user via GET /api/me.

        Returns:
            The user object returned by the server.
        """
        self._user = self._request("GET", "/api/me")
        return self._user

    def record_type_schema(self, record_type: str) -> dict[str, Any]:
        """Fetch the declarative EAV field schema for a record_type via
        GET /api/record-types/{recordType}/schema.

        This is what `o2org.pipelines.utils.extract.extract()` uses to
        flatten `details` JSON without hardcoding a Python copy of the
        server's `ExperimentRecord::fieldSchema()`.

        Args:
            record_type: e.g. "result_stable_isotopes".

        Returns:
            `{"record_type", "label", "family", "fields": [...]}`.
        """
        return self._request("GET", f"/api/record-types/{record_type}/schema")

    def _set_token(self, token: str) -> None:
        self._token = token
        self.session.headers["Authorization"] = f"Bearer {token}"

    @property
    def is_authenticated(self) -> bool:
        """Whether a bearer token is currently set (does not verify it's still valid server-side)."""
        return self._token is not None

    def __enter__(self) -> OrgClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.logout()

    def __repr__(self) -> str:
        who = self._email or "token-auth"
        return f"<OrgClient {self.base_url} as {who}>"

    # -- generic verbs, used by resources.py ---------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Send an authenticated GET request.

        Args:
            path: API path, e.g. "/api/compounds".
            params: Query-string parameters.

        Returns:
            The parsed JSON response body, or None for an empty/204 response.
        """
        return self._request("GET", path, params=params)

    def post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        """Send an authenticated POST request with a JSON body.

        Args:
            path: API path, e.g. "/api/compounds".
            json: Request body, serialized as JSON.

        Returns:
            The parsed JSON response body, or None for an empty/204 response.
        """
        return self._request("POST", path, json=json)

    def put(self, path: str, json: dict[str, Any] | None = None) -> Any:
        """Send an authenticated PUT request with a JSON body.

        Args:
            path: API path, e.g. "/api/compounds/42".
            json: Request body, serialized as JSON.

        Returns:
            The parsed JSON response body, or None for an empty/204 response.
        """
        return self._request("PUT", path, json=json)

    def patch(self, path: str, json: dict[str, Any] | None = None) -> Any:
        """Send an authenticated PATCH request with a JSON body.

        Args:
            path: API path, e.g. "/api/compounds/42".
            json: Request body, serialized as JSON.

        Returns:
            The parsed JSON response body, or None for an empty/204 response.
        """
        return self._request("PATCH", path, json=json)

    # -- plumbing -------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, auth_required: bool = True, **kwargs: Any) -> Any:
        if auth_required and not self.is_authenticated:
            raise AuthenticationError(
                "Not authenticated. Call connect(email=..., password=...) or "
                "client.login(email, password) first."
            )
        url = self._url(path)
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.exceptions.RequestException as exc:
            raise APIError(f"Could not reach {url}: {exc}") from exc

        if response.status_code == 401:
            raise AuthenticationError(
                f"{method} {path} -> 401 Unauthorized. Token missing or expired; "
                "call login() again."
            )
        if response.status_code == 404:
            raise NotFoundError(f"{method} {path} -> 404 Not Found", status_code=404)
        if response.status_code == 422:
            raise ValidationError(
                f"{method} {path} -> 422 Unprocessable: {response.text}",
                status_code=422, payload=_safe_json(response),
            )
        if not response.ok:
            raise APIError(
                f"{method} {path} -> {response.status_code}: {response.text}",
                status_code=response.status_code, payload=_safe_json(response),
            )

        if response.status_code == 204 or not response.content:
            return None
        return response.json()
