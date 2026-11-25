import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from api.deps import RedisRateLimiter


def test_redis_rate_limiter_hashes_api_key_and_rejects_excess_requests(monkeypatch):
    client = Mock()
    client.eval.return_value = 0
    redis_module = SimpleNamespace(
        Redis=SimpleNamespace(from_url=Mock(return_value=client)),
    )
    monkeypatch.setitem(sys.modules, "redis", redis_module)
    limiter = RedisRateLimiter("redis://localhost:6379/0", max_requests=1, window_seconds=60)

    with pytest.raises(HTTPException, match="Rate limit exceeded") as error:
        limiter.check("sensitive-api-key")

    assert error.value.status_code == 429
    assert "sensitive-api-key" not in str(client.eval.call_args)
    expected_key = "verimind:rate-limit:" + hashlib.sha256(b"sensitive-api-key").hexdigest()
    assert client.eval.call_args.args[2] == expected_key
