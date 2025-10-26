from dataclasses import dataclass
import importlib

from src.policy.env_guard import enforce_env_guard


def set_mode(monkeypatch, mode: str, kill: str | None = None):
    monkeypatch.setenv("MODE", mode)
    if kill is None:
        monkeypatch.delenv("KILL_SWITCH", raising=False)
    else:
        monkeypatch.setenv("KILL_SWITCH", kill)
    # Refresh safety module (though trading_enabled reads env on call)
    import src.core.safety as safety

    importlib.reload(safety)


def test_env_guard_on_dict(monkeypatch):
    set_mode(monkeypatch, "paper")  # force disabled
    decision = {"allow": True, "reasons": []}
    out = enforce_env_guard(decision)
    assert out["allow"] is False
    rs = out.get("reasons", [])
    assert any("MODE" in r or "KILL" in r or "disabled" in r for r in rs)


def test_env_guard_on_dataclass(monkeypatch):
    set_mode(monkeypatch, "live", kill="1")  # disabled by kill switch

    @dataclass
    class D:
        allow: bool
        score: float = 0.0

    d = D(allow=True, score=0.7)
    out = enforce_env_guard(d)
    assert isinstance(out, D)
    assert out.allow is False
    assert out.score == 0.7
