import pandas as pd
import importlib


def set_mode(monkeypatch, mode: str, kill: str | None = None):
    monkeypatch.setenv("MODE", mode)
    if kill is None:
        monkeypatch.delenv("KILL_SWITCH", raising=False)
    else:
        monkeypatch.setenv("KILL_SWITCH", kill)
    # Reload safety to ensure any module-level caching would pick new env
    import src.core.safety as safety

    importlib.reload(safety)


def test_df_guard_disables_when_paper(monkeypatch):
    set_mode(monkeypatch, "paper")
    from src.policy.df_guard import enforce_env_guard_df

    df = pd.DataFrame({"signal": [0.7, 0.8], "trade_ok": [True, True]})
    out = enforce_env_guard_df(df)
    assert out["trade_ok"].eq(False).all()
    assert "deny_reason" in out.columns
    msgs = out["deny_reason"].astype(str)
    assert (msgs.str.contains("MODE", case=False) | msgs.str.contains("KILL", case=False) | msgs.str.contains("disabled", case=False)).all()


def test_df_guard_no_trade_ok_column(monkeypatch):
    set_mode(monkeypatch, "paper")
    from src.policy.df_guard import enforce_env_guard_df

    df = pd.DataFrame({"signal": [0.2, 0.9]})
    out = enforce_env_guard_df(df)
    assert out["trade_ok"].eq(False).all()
    assert "deny_reason" in out.columns


def test_df_guard_noop_when_live(monkeypatch):
    set_mode(monkeypatch, "live")
    from src.policy.df_guard import enforce_env_guard_df

    df = pd.DataFrame({"signal": [0.2, 0.9], "trade_ok": [True, False]})
    out = enforce_env_guard_df(df)
    # Unchanged
    assert out.equals(df)
