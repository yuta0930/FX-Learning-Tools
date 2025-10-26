import pandas as pd
from datetime import datetime, timedelta


def test_spread_filter_basic():
    from src.policy.filters import apply_spread_filter

    df = pd.DataFrame({"spread_pips": [0.2, 1.5, 0.4], "trade_ok": [True, True, True]})
    out = apply_spread_filter(df, max_spread_pips=1.0)
    assert out["trade_ok"].tolist() == [True, False, True]
    assert "deny_reason" in out.columns
    assert "spread>1.0" in str(out.loc[1, "deny_reason"])


def test_atr_filter_min_max():
    from src.policy.filters import apply_atr_filter

    df = pd.DataFrame({"atr": [0.05, 0.8, 3.0], "trade_ok": [True, True, True]})
    out = apply_atr_filter(df, min_atr=0.1, max_atr=2.0)
    assert out["trade_ok"].tolist() == [False, True, False]
    assert "atr<0.1" in str(out.loc[0, "deny_reason"])
    assert "atr>2.0" in str(out.loc[2, "deny_reason"])


def test_news_window_filter():
    from src.policy.filters import apply_news_window_filter

    base = datetime(2025, 1, 1, 9, 0, 0)
    df = pd.DataFrame({
        "time": [base + timedelta(minutes=m) for m in [0, 10, 40]],
        "trade_ok": [True, True, True],
    })
    events = pd.DataFrame({"time": [base + timedelta(minutes=15)]})
    out = apply_news_window_filter(df, events=events, minutes_before=10, minutes_after=10)
    # 9:10〜9:20がウィンドウ → 10分の行(9:10)だけ不可
    assert out["trade_ok"].tolist() == [True, False, True]
    assert "news_window" in str(out.loc[1, "deny_reason"])
