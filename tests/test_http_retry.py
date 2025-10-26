import httpx

from src.net.http import request_with_retry


class DummyResp:
    def __init__(self, status_code: int):
        self.status_code = status_code


def test_retry_on_status():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] < 3:
            return DummyResp(503)  # retry
        return DummyResp(200)

    out = request_with_retry(call, retries=3, sleeper=lambda s: None)
    assert out.status_code == 200
    assert calls["n"] == 3


def test_retry_on_exception():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ReadTimeout("timeout")
        return DummyResp(200)

    out = request_with_retry(call, retries=3, sleeper=lambda s: None)
    assert out.status_code == 200
    assert calls["n"] == 2
