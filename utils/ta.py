# -*- coding: utf-8 -*-
"""
Common TA utilities for FX-Learning-Tools.
- Pure functions: no streamlit/session side-effects
- Inputs expect lower-case columns: open/high/low/close[/volume]
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional

__all__ = [
    "atr",
    "swing_pivots",
    "horizontal_levels",
    "regression_trend",
]

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (simple moving average).
    Requirements: df has columns high, low, close (lower-case)
    Returns: pd.Series aligned to df.index
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series([], dtype=float)
    need = {"high", "low", "close"}
    cols = {c.lower() for c in df.columns}
    if not need.issubset(cols):
        raise ValueError(f"atr requires columns: {need}")
    d = df.rename(columns=str.lower)
    high, low, close = d["high"], d["low"], d["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(int(period), min_periods=int(period)).mean()

def swing_pivots(df: pd.DataFrame, look: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect swing highs/lows using centered rolling window of size=look.
    Returns: (pivot_high_df, pivot_low_df)
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = df.rename(columns=str.lower)
    highs = d["high"].rolling(int(look), center=True).max()
    lows  = d["low"].rolling(int(look),  center=True).min()
    ph = d[d["high"] == highs].dropna(subset=["high"]).copy()
    pl = d[d["low"]  == lows ].dropna(subset=["low"]).copy()
    return ph, pl

def horizontal_levels(pivot_high: pd.DataFrame, pivot_low: pd.DataFrame, *, eps: Optional[float], min_samples: int = 4) -> List[float]:
    """Cluster pivot prices horizontally (DBSCAN-like behavior via sklearn if available).
    When sklearn is missing, fallback to simple binning by eps.
    """
    import math
    prices = []
    if isinstance(pivot_high, pd.DataFrame) and not pivot_high.empty and "high" in pivot_high.columns:
        prices.extend(pivot_high["high"].values.tolist())
    if isinstance(pivot_low, pd.DataFrame) and not pivot_low.empty and "low" in pivot_low.columns:
        prices.extend(pivot_low["low"].values.tolist())
    if not prices:
        return []
    arr = np.asarray(prices, dtype=float).reshape(-1, 1)
    levels: List[float] = []
    try:
        from sklearn.cluster import DBSCAN
        auto_eps = float(np.std(arr)) * 0.05 if not eps or eps <= 0 else float(eps)
        ms = max(3, int(min_samples))
        labels = DBSCAN(eps=auto_eps, min_samples=ms).fit(arr).labels_
        for lab in set(labels) - {-1}:
            lv = arr[labels == lab].mean()
            levels.append(float(lv))
    except Exception:
        # Fallback: simple bucketing by eps
        auto_eps = float(np.std(arr)) * 0.05 if not eps or eps <= 0 else float(eps)
        bins = {}
        for v in arr.flatten():
            key = round(v / auto_eps) if auto_eps > 0 else 0
            bins.setdefault(key, []).append(float(v))
        for key, vals in bins.items():
            if len(vals) >= max(3, int(min_samples)):
                levels.append(float(np.mean(vals)))
    # de-duplicate close levels
    levels = sorted(set([round(v, 3) for v in levels]))
    return levels

def regression_trend(df: pd.DataFrame, lookback: int, use: str = "low") -> Optional[Dict[str, float]]:
    """Simple linear regression over last N points.
    Returns dict with slope/intercept/sigma and line endpoints, or None if insufficient data.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    d = df.rename(columns=str.lower)
    sub = d.tail(int(lookback))
    if len(sub) < 2 or use not in sub.columns:
        return None
    y = sub[use].astype(float).values
    x = np.arange(len(y), dtype=float)
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    resid = y - yhat
    sigma = float(np.std(resid))
    t0, t1 = sub.index[0], sub.index[-1]
    y0, y1 = float(b), float(m * (len(x) - 1) + b)
    return dict(x0=t0, y0=y0, x1=t1, y1=y1, slope=float(m), intercept=float(b), sigma=sigma, n=int(len(x)))
