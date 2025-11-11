import pandas as pd
import numpy as np

from features_util import compute_level_relative_features

def make_df(n=300, start=150.0, step=0.02):
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = start + np.arange(n) * step
    # simple OHLC around close
    high = close + 0.05
    low  = close - 0.05
    open_ = close - 0.01
    vol = np.ones(n)
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })


def test_level_features_basic():
    df = make_df(240)
    out = compute_level_relative_features(df, window=120, look_pivot=9, k=2)
    # Columns exist
    need = ["lvl_k1_up","lvl_k1_dn","lvl_k2_up","lvl_k2_dn","lvl_near_cnt","lvl_near_flag","lvl_span_atr"]
    for c in need:
        assert c in out.columns
    # lengths
    assert len(out) == len(df)
    # finite
    assert np.isfinite(out[need].values).all()
    # non-negative metrics
    assert (out[["lvl_k1_up","lvl_k1_dn","lvl_near_cnt","lvl_span_atr"]] >= 0).all().all()


def test_level_features_near_detects():
    # Craft a plateau level around 151.0 in the last window
    n = 200
    df = make_df(n)
    # Overwrite last 60 bars to hover around 151.0 to induce pivots/levels
    base = 151.0
    for i in range(n-60, n):
        df.loc[i, "close"] = base + (0.001 if (i % 2 == 0) else -0.001)
        df.loc[i, "high"]  = df.loc[i, "close"] + 0.01
        df.loc[i, "low"]   = df.loc[i, "close"] - 0.01
    out = compute_level_relative_features(df, window=120, look_pivot=7, k=2)
    # The last bar should be near some level
    assert float(out["lvl_near_cnt"].iloc[-1]) >= 1.0
