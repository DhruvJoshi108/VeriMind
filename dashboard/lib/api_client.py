"""Typed, resilient HTTP boundary for the dashboard."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ApiError:
    message: str
    request_id: str
    status_code: int | None = None
    retry_after: str | None = None


@dataclass(frozen=True)
class ApiResult:
    data: dict[str, Any] | str | None = None
    error: ApiError | None = None
    request_id: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


class ApiClient:
    """HTTP client that never leaks requests exceptions to Streamlit pages."""

    def __init__(self, base_url: str, api_key: str = "", timeout_seconds: float = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def _request(
        self, method: str, path: str, *, requires_api_key: bool = False, **kwargs: Any
    ) -> ApiResult:
        request_id = str(uuid.uuid4())
        headers = {"X-Request-ID": request_id, **kwargs.pop("headers", {})}
        if requires_api_key and self.api_key:
            headers["x-api-key"] = self.api_key
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                timeout=self.timeout_seconds,
                **kwargs,
            )
        except requests.Timeout:
            return ApiResult(
                error=ApiError("The API timed out. Please try again.", request_id), request_id=request_id
            )
        except requests.ConnectionError:
            return ApiResult(
                error=ApiError("The API is unavailable. Check the URL and service status.", request_id),
                request_id=request_id,
            )
        except requests.RequestException:
            return ApiResult(
                error=ApiError("The API request could not be completed.", request_id), request_id=request_id
            )

        returned_id = response.headers.get("x-request-id") or request_id
        if response.status_code >= 400:
            messages = {
                401: "Authentication failed. Check the API key.",
                403: "You are not allowed to use this API.",
                422: "The claim is invalid or exceeds the API limit.",
                429: "Rate limit reached. Please wait before trying again.",
            }
            message = messages.get(response.status_code)
            if message is None:
                message = (
                    "The API had an internal error."
                    if response.status_code >= 500
                    else "The API rejected the request."
                )
            return ApiResult(
                error=ApiError(message, returned_id, response.status_code, response.headers.get("Retry-After")),
                request_id=returned_id,
            )
        try:
            payload = response.json()
        except ValueError:
            if path == "/metrics":
                return ApiResult(data=response.text, request_id=returned_id)
            else:
                return ApiResult(
                    error=ApiError("The API returned malformed JSON.", returned_id, response.status_code),
                    request_id=returned_id,
                )
        if not isinstance(payload, dict):
            return ApiResult(
                error=ApiError("The API returned an unexpected response.", returned_id, response.status_code),
                request_id=returned_id,
            )
        return ApiResult(data=payload, request_id=returned_id)

    def health(self) -> ApiResult:
        return self._request("GET", "/healthz")

    def ready(self) -> ApiResult:
        return self._request("GET", "/readyz")

    def metrics(self) -> ApiResult:
        return self._request("GET", "/metrics")

    def verify(self, text: str) -> ApiResult:
        return self._request("POST", "/verify", requires_api_key=True, json={"text": text})
