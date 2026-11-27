from unittest.mock import patch

import pytest

from isofood_center.client import OrgClient
from isofood_center.exceptions import AuthenticationError, NotFoundError, ValidationError

from .helpers import FakeResponse


def test_login_sets_bearer_token():
    login_response = FakeResponse(200, {"token": "abc123", "user": {"email": "a@b.com"}})
    with patch("requests.Session.request", return_value=login_response) as mock_request:
        client = OrgClient("https://dma.example.org", email="a@b.com", password="secret")

    assert client.is_authenticated
    assert client.session.headers["Authorization"] == "Bearer abc123"
    mock_request.assert_called_once()
    method, url = mock_request.call_args.args[:2]
    assert method == "POST"
    assert url == "https://dma.example.org/api/login"


def test_login_without_token_field_raises():
    login_response = FakeResponse(200, {"user": {}})
    with patch("requests.Session.request", return_value=login_response):
        with pytest.raises(AuthenticationError):
            OrgClient("https://dma.example.org", email="a@b.com", password="secret")


def test_request_without_auth_raises_before_any_http_call():
    client = OrgClient.__new__(OrgClient)  # bypass __init__/login entirely
    client.base_url = "https://dma.example.org"
    client.timeout = 30
    client._token = None
    import requests
    client.session = requests.Session()

    with pytest.raises(AuthenticationError):
        client.get("/api/compounds")


def test_401_raises_authentication_error():
    with patch("requests.Session.request", return_value=FakeResponse(200, {"token": "t"})):
        client = OrgClient("https://dma.example.org", token="preset-token")
    with patch("requests.Session.request", return_value=FakeResponse(401, text="unauthorized")):
        with pytest.raises(AuthenticationError):
            client.get("/api/compounds")


def test_404_raises_not_found_error():
    client = OrgClient("https://dma.example.org", token="preset-token")
    with patch("requests.Session.request", return_value=FakeResponse(404, text="missing")):
        with pytest.raises(NotFoundError):
            client.get("/api/compounds/999")


def test_422_raises_validation_error_with_payload():
    client = OrgClient("https://dma.example.org", token="preset-token")
    errors = {"errors": {"canonical_name": ["required"]}}
    with patch("requests.Session.request", return_value=FakeResponse(422, errors)):
        with pytest.raises(ValidationError) as exc_info:
            client.post("/api/compounds", json={})
    assert exc_info.value.payload == errors


def test_logout_clears_token():
    client = OrgClient("https://dma.example.org", token="preset-token")
    with patch("requests.Session.request", return_value=FakeResponse(204)):
        client.logout()
    assert not client.is_authenticated
    assert "Authorization" not in client.session.headers


def test_record_type_schema_hits_schema_endpoint():
    client = OrgClient("https://dma.example.org", token="preset-token")
    payload = {"record_type": "result_stable_isotopes", "label": "Result — stable isotopes", "family": "result", "fields": []}
    with patch("requests.Session.request", return_value=FakeResponse(200, payload)) as mock_request:
        schema = client.record_type_schema("result_stable_isotopes")
    method, url = mock_request.call_args.args[:2]
    assert method == "GET"
    assert url == "https://dma.example.org/api/record-types/result_stable_isotopes/schema"
    assert schema == payload


def test_context_manager_logs_out():
    with patch("requests.Session.request", return_value=FakeResponse(204)):
        with OrgClient("https://dma.example.org", token="preset-token") as client:
            assert client.is_authenticated
        assert not client.is_authenticated
