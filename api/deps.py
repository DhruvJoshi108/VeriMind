"""API authentication, lazy pipeline access, and thread-safe rate limiting."""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict, deque
from threading import Lock
from time import time, time_ns
from typing import Any

from fastapi import HTTPException, Request

from verimind.config import settings
from verimind.pipeline import VerificationPipeline

_pipeline_lock = Lock()


class InMemoryRateLimiter:
    """Sliding-window fallback. Use a gateway/shared store for multi-worker production."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, api_key: str) -> None:
        with self._lock:
            now = time()
            bucket = self.requests[api_key]
            while bucket and now - bucket[0] >= self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            bucket.append(now)


class RedisRateLimiter:
    """Atomic shared sliding-window limiter for multi-instance deployments."""

    _script = """
    local now = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, now - window)
    if redis.call('ZCARD', KEYS[1]) >= limit then
        return 0
    end
    redis.call('ZADD', KEYS[1], now, ARGV[4])
    redis.call('PEXPIRE', KEYS[1], window)
    return 1
    """

    def __init__(self, url: str, max_requests: int, window_seconds: int) -> None:
        import redis

        self.client = redis.Redis.from_url(url, socket_connect_timeout=5, socket_timeout=5)
        self.max_requests = max_requests
        self.window_ms = window_seconds * 1_000

    def check(self, api_key: str) -> None:
        key = "verimind:rate-limit:" + hashlib.sha256(api_key.encode()).hexdigest()
        allowed = self.client.eval(
            self._script,
            1,
            key,
            time_ns() // 1_000_000,
            self.window_ms,
            self.max_requests,
            uuid.uuid4().hex,
        )
        if not allowed:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")


rate_limiter = (
    RedisRateLimiter(
        settings.redis_url,
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
    )
    if settings.redis_url
    else InMemoryRateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)
)


def get_api_key(request: Request) -> str:
    """Validate x-api-key; development allows an explicit or default demo key."""
    api_key = request.headers.get("x-api-key")
    if not settings.api_key:
        if settings.environment == "production":
            raise HTTPException(status_code=503, detail="API authentication is not configured")
        return api_key or "demo-key"
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


def require_api_key(request: Request) -> str:
    api_key = get_api_key(request)
    rate_limiter.check(api_key)
    return api_key


def get_pipeline(request: Request) -> VerificationPipeline:
    """Create the expensive pipeline only on its first verification request."""
    pipeline: Any = getattr(request.app.state, "pipeline", None)
    if pipeline is not None:
        return pipeline
    with _pipeline_lock:
        pipeline = getattr(request.app.state, "pipeline", None)
        if pipeline is None:
            pipeline = VerificationPipeline()
            request.app.state.pipeline = pipeline
    return pipeline
