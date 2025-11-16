import pandas as pd
from datetime import datetime, timedelta
from src.policy.pipeline import run_market_filters, FilterConfig
from src.core.market_profile import MarketProfile
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


def test_pipeline_respects_market_profile_atr_override(monkeypatch):
    set_mode(monkeypatch, "live")
    base = datetime(2025, 1, 1, 9, 0, 0)
    df = pd.DataFrame(
        {
            "time": [base + timedelta(minutes=m) for m in [0, 10, 20]],
            "open": [150.0, 150.1, 150.2],
            "high": [150.2, 150.3, 150.4],
            "low": [149.9, 150.0, 150.1],
            "close": [150.1, 150.2, 150.3],
            "bid": [150.10, 150.12, 150.15],
            "ask": [150.11, 150.13, 150.16],
            "atr": [0.6, 1.1, 1.4],
        }
    )

    cfg = FilterConfig(max_spread_pips=5.0, atr_min=0.0, atr_max=5.0)
    profile = MarketProfile(session="Tokyo", atr_regime="mid", latest_atr=None)
    auto_cfg = {
        "profiles": {
            "Tokyo": {
                "mid": {
                    "atr_filter": {"min_atr": 1.0, "max_atr": 2.0},
                }
            }
        }
    }

    baseline = run_market_filters(df, cfg=cfg)
    overridden = run_market_filters(
        df,
        cfg=cfg,
        auto_params=auto_cfg,
        market_profile=profile,
    )

    assert baseline["trade_ok"].sum() == 3
    assert overridden["trade_ok"].sum() == 2
    assert not overridden.loc[overridden["atr"] < 1.0, "trade_ok"].any()
