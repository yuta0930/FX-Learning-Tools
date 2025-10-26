import importlib
from pathlib import Path


def reload_safety():
    import src.core.safety as safety

    importlib.reload(safety)
    return safety


def test_paper_mode_disables_trading(monkeypatch):
    monkeypatch.setenv("MODE", "paper")
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    safety = reload_safety()
    assert safety.current_mode() == "paper"
    assert safety.trading_enabled() is False


def test_kill_switch_env_overrides(monkeypatch):
    monkeypatch.setenv("MODE", "live")
    monkeypatch.setenv("KILL_SWITCH", "1")
    safety = reload_safety()
    assert safety.is_kill_switch_on() is True
    assert safety.trading_enabled() is False


def test_kill_switch_file_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("MODE", "live")
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    import src.core.safety as safety

    # Redirect flags dir to a temporary location for the test
    tmp_flags = tmp_path / "flags"
    tmp_flags.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(safety, "FLAGS_DIR", tmp_flags, raising=False)
    monkeypatch.setattr(safety, "KILL_FILE", tmp_flags / "kill.switch", raising=False)

    safety.KILL_FILE.write_text("on", encoding="utf-8")

    # No reload needed; functions reference module globals we just patched
    assert safety.is_kill_switch_on() is True
    assert safety.trading_enabled() is False
