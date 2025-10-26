from __future__ import annotations

from typing import Callable, Optional
import time
import os
import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504}


def request_with_retry(
    call: Callable[[], httpx.Response],
    *,
    retries: int = int(os.getenv("HTTP_MAX_RETRIES", "3")),
    timeout_s: float = float(os.getenv("HTTP_TIMEOUT_SECONDS", "10")),
    backoff_ms: int = int(os.getenv("HTTP_BACKOFF_BASE_MS", "200")),
    sleeper: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = call()
            if hasattr(resp, "status_code") and resp.status_code in RETRY_STATUSES and attempt < retries:
                sleeper(backoff_ms / 1000.0 * (2**attempt))
                continue
            return resp
        except (httpx.TransportError, httpx.ReadTimeout) as e:
            last_exc = e
            if attempt < retries:
                sleeper(backoff_ms / 1000.0 * (2**attempt))
                continue
            raise

    if last_exc:
        raise last_exc
    raise RuntimeError("request_with_retry: unexpected flow")


def get(url: str, **kwargs) -> httpx.Response:
    timeout_s = float(os.getenv("HTTP_TIMEOUT_SECONDS", "10"))
    with httpx.Client(timeout=timeout_s) as client:
        return request_with_retry(lambda: client.get(url, **kwargs))
