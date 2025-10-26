import pandas as pd
from datetime import datetime, timedelta
from src.policy.pipeline import run_market_filters, FilterConfig
import importlib


def set_mode(monkeypatch, mode="paper", kill=None):
    monkeypatch.setenv("MODE", mode)
    if kill is None:
        monkeypatch.delenv("KILL_SWITCH", raising=False)
    else:
        monkeypatch.setenv("KILL_SWITCH", str(kill))
    # reload safety (defensive; current implementation reads env dynamically)
    import src.core.safety as safety

    importlib.reload(safety)


def test_pipeline_basic(monkeypatch):
    # Expect environment guard to disable all trades in paper mode
    set_mode(monkeypatch, "paper")
    base = datetime(2025, 1, 1, 9, 0, 0)
    df = pd.DataFrame(
        {
            "time": [base + timedelta(minutes=m) for m in [0, 10, 40]],
            "open": [150.0, 150.1, 150.2],
            "high": [150.2, 150.4, 150.5],
            "low": [149.9, 150.0, 150.1],
            "close": [150.1, 150.2, 150.3],
            "bid": [150.10, 150.20, 150.30],
            "ask": [150.11, 150.25, 150.31],  # wider spread on middle row (~5 pips)
        }
    )
    events = pd.DataFrame({"time": [base + timedelta(minutes=15)]})

    cfg = FilterConfig(
        max_spread_pips=3.0,
        atr_min=0.0,
        atr_max=5.0,
        news_before_min=10,
        news_after_min=10,
    )
    out = run_market_filters(df, events_df=events, cfg=cfg)

    assert "trade_ok" in out.columns
    assert out["trade_ok"].eq(False).all()
    assert "deny_reason" in out.columns
