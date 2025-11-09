from __future__ import annotations

import numpy as np
import pandas as pd

from src.patterns.flag_pennant import detect_flag_pennant_simplified
from src.core.ta import add_atr_if_missing


def _make_series_flag_long(n_pole: int = 20, n_cons: int = 12, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # base flat segment
    base = np.full(40, 100.0)
    # impulsive pole up
    pole = 100.0 + np.cumsum(np.full(n_pole, 0.25))  # +0.25 per bar
    # consolidation: slight downward drift with small noise
    cons_center = pole[-1]
    cons = cons_center + np.linspace(0.0, -0.5, n_cons) + rng.normal(0.0, 0.05, n_cons)
    # breakout bar up
    brk = np.array([cons[-1] + 0.6])

    close = np.concatenate([base, pole, cons, brk])
    # build OHLC around close
    high = close + 0.05
    low = close - 0.05
    open_ = close.copy()

    idx = pd.date_range("2024-01-01", periods=len(close), freq="15min", tz="UTC")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    df = add_atr_if_missing(df)
    return df


def test_detect_flag_pennant_simplified_long_breakout():
    df = _make_series_flag_long()
    # Use lenient thresholds to ensure detection on synthetic data
    sigs = detect_flag_pennant_simplified(
        df,
        lookback=500,
        n_push=18,
        min_flag_bars=8,
        max_flag_bars=20,
        sigma_k=1.0,
        pole_min_atr=1.5,
        flag_slope_max_atr=0.2,
        contraction_percentile=0.5,
    )
    assert isinstance(sigs, list)
    assert len(sigs) >= 1, "No flag/pennant detected on synthetic long pattern"
    # Check latest signal properties
    s = sigs[-1]
    assert s.side == 1
    assert s.kind in ("flag", "pennant")
    assert 0.0 <= s.quality <= 1.0
