import pandas as pd
import numpy as np
from src.ui.features_cache import prepare_feats_cached

def _make_df(n=200):
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    base = np.cumsum(np.random.randn(n)) + 150
    df = pd.DataFrame({
        "timestamp": ts,
        "open": base + np.random.randn(n)*0.1,
        "high": base + 0.3 + np.random.randn(n)*0.1,
        "low": base - 0.3 + np.random.randn(n)*0.1,
        "close": base + np.random.randn(n)*0.1,
        "volume": np.random.randint(100, 200, size=n)
    })
    return df


def test_prepare_feats_cached_basic():
    raw = _make_df()
    feats = prepare_feats_cached(raw)
    assert "timestamp" in feats.columns
    assert len(feats) > 0


def test_prepare_feats_cached_changes_invalidate():
    raw = _make_df()
    feats1 = prepare_feats_cached(raw)
    # Slight perturbation to raw should invalidate (different content)
    raw2 = raw.copy()
    raw2.loc[10, "close"] += 1.23
    feats2 = prepare_feats_cached(raw2)
    # Expect a difference in at least one column aggregate
    assert float(feats1["close"].mean()) != float(feats2["close"].mean())
