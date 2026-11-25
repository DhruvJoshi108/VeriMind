from unittest.mock import Mock

import requests

from dashboard.lib.api_client import ApiClient


def response(status: int, body=None, headers=None):
    item = Mock(status_code=status, headers=headers or {})
    if isinstance(body, Exception):
        item.json.side_effect = body
    else:
        item.json.return_value = body
    item.text = "not-json"
    return item


def test_verify_success_sends_request_id_and_key():
    client = ApiClient("http://api", "secret")
    client.session.request = Mock(return_value=response(200, {"claims": []}, {"x-request-id": "server-id"}))

    result = client.verify("Paris is in France.")

    assert result.ok
    assert result.data == {"claims": []}
    assert result.request_id == "server-id"
    headers = client.session.request.call_args.kwargs["headers"]
    assert headers["x-api-key"] == "secret"
    assert headers["X-Request-ID"]


def test_health_and_readiness_do_not_send_api_key_but_metrics_does():
    client = ApiClient("http://api", "secret")
    client.session.request = Mock(
        side_effect=[
            response(200, {"status": "ok"}),
            response(200, {"status": "ready"}),
            response(200, ValueError("plain text")),
        ]
    )

    assert client.health().ok
    assert client.ready().ok
    assert client.metrics().ok
    assert "x-api-key" not in client.session.request.call_args_list[0].kwargs["headers"]
    assert "x-api-key" not in client.session.request.call_args_list[1].kwargs["headers"]
    assert client.session.request.call_args_list[2].kwargs["headers"]["x-api-key"] == "secret"


def test_timeout_is_a_friendly_error():
    client = ApiClient("http://api")
    client.session.request = Mock(side_effect=requests.Timeout())

    assert "timed out" in client.verify("claim").error.message


def test_rate_limit_exposes_retry_after():
    client = ApiClient("http://api")
    client.session.request = Mock(return_value=response(429, {}, {"Retry-After": "12"}))

    result = client.verify("claim")

    assert result.error.status_code == 429
    assert result.error.retry_after == "12"


def test_server_error_and_malformed_json_are_friendly():
    client = ApiClient("http://api")
    client.session.request = Mock(return_value=response(503, {}))
    assert "internal error" in client.verify("claim").error.message
    client.session.request = Mock(return_value=response(200, ValueError("bad json")))
    assert "malformed JSON" in client.verify("claim").error.message


def test_authentication_validation_and_unexpected_json_are_friendly():
    client = ApiClient("http://api")
    for status, expected in ((401, "Authentication failed"), (403, "not allowed"), (422, "invalid")):
        client.session.request = Mock(return_value=response(status, {}))
        assert expected in client.verify("claim").error.message

    client.session.request = Mock(return_value=response(200, ["not", "an object"]))
    result = client.verify("claim")
    assert not result.ok
    assert "unexpected response" in result.error.message
