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
    """Average True Range using Wilder's smoothing (EWM, alpha=1/period).
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
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/float(period), adjust=False).mean()

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

def horizontal_levels(
    pivot_high: pd.DataFrame,
    pivot_low: pd.DataFrame,
    *,
    eps: Optional[float],
    min_samples: int = 4,
    line_min_hits: Optional[int] = None,
    round_step: Optional[float] = None,
    merge_near: Optional[float] = None,
    quality_out: Optional[Dict[str, float]] = None,
    cache: bool = True,
) -> List[float]:
    """Cluster pivot prices horizontally with robust defaults and small-cache.

    - If eps is None/<=0: auto-eps via robust scale (IQR) fallback to std.
        - Quantize prices to step grid (guessed pip/2 by default) BEFORE clustering
            to stabilize behavior across min_samples changes.
        - Merge adjacent levels within merge_near (default ≈ pip). Merge threshold
            is at least the step size to avoid jitter-driven splits.
    - Optionally record quality metrics to quality_out dict.
    - Lightweight in-process cache keyed by simple summaries.
    """
    import math

    def _robust_scale(x: np.ndarray) -> float:
        q75, q25 = np.percentile(x, [75, 25])
        iqr = float(q75 - q25)
        s = float(np.std(x))
        # Prefer IQR; fallback to std; avoid zero
        base = iqr if iqr > 0 else s
        return base if base > 0 else (float(np.mean(np.abs(x - np.mean(x)))) or 1.0)

    def _guess_pip(x: np.ndarray) -> float:
        med = float(np.median(x)) if x.size else 1.0
        # Very rough: JPY pairs (~>10) use 0.01 pip, otherwise 0.0001
        return 0.01 if med >= 10 else 0.0001

    def _round_to_step(val: float, step: float) -> float:
        if step <= 0: return float(val)
        return float(round(val / step) * step)

    # Collect prices
    prices: List[float] = []
    if isinstance(pivot_high, pd.DataFrame) and not pivot_high.empty and "high" in pivot_high.columns:
        prices.extend(pd.to_numeric(pivot_high["high"], errors="coerce").dropna().astype(float).tolist())
    if isinstance(pivot_low, pd.DataFrame) and not pivot_low.empty and "low" in pivot_low.columns:
        prices.extend(pd.to_numeric(pivot_low["low"], errors="coerce").dropna().astype(float).tolist())
    if not prices:
        if isinstance(quality_out, dict):
            quality_out.update(dict(n_prices=0, n_levels_raw=0, n_levels_merged=0, eps_used=float(eps or 0.0)))
        return []

    arr = np.asarray(prices, dtype=float).reshape(-1, 1)
    # Cluster min_samples (DBSCAN density).
    # 以前は安定性のために下限=3に強制していたが、テスト/用途に応じて1や2も許容する。
    # （デフォルト min_samples=4 で従来挙動は維持）
    ms = max(1, int(min_samples))
    # line_min_hits: DBSCANクラスタから水平線として採用するための最小ヒット数（未指定時は min_samples と同一）
    # Line adoption threshold: if None, fall back to cluster min_samples (legacy behavior)
    lm_hits = int(line_min_hits) if (line_min_hits is not None and int(line_min_hits) > 0) else ms

    # Auto eps if needed (robust)
    auto_scale = _robust_scale(arr.flatten())
    eps_used = float(eps) if (eps is not None and eps > 0) else max(1e-9, auto_scale * 0.05)

    # Determine step/marge thresholds up-front (affects cache & stability)
    if round_step is None:
        step = _guess_pip(arr.flatten()) / 2.0
    else:
        step = float(round_step)
    if merge_near is None:
        mthr = _guess_pip(arr.flatten())
    else:
        mthr = float(merge_near)
    # Quantize prices to the step grid BEFORE clustering to reduce
    # non-monotonic effects from later rounding/merging.
    def _q(val: float) -> float:
        return _round_to_step(val, step)
    qarr_flat = np.array([_q(v) for v in arr.flatten()], dtype=float)
    qarr = qarr_flat.reshape(-1, 1)

    # Cache key (summaries only; not perfect but effective)
    cache_hit = False
    if cache:
        # Include line_min_hits in cache key to avoid returning clusters filtered differently.
        key = (
            len(qarr),
            float(qarr[-1][0]),
            round(eps_used, 8),
            ms,
            lm_hits,
            round(float(step), 8),
            round(float(mthr), 8),
        )
        store = getattr(horizontal_levels, "_cache_store", None)
        if store is None:
            horizontal_levels._cache_store = {}
            store = horizontal_levels._cache_store
        if key in store:
            cache_hit = True
            levels_cached = store[key]
            if isinstance(quality_out, dict):
                quality_out.update(dict(n_prices=len(qarr), n_levels_raw=len(levels_cached), n_levels_merged=len(levels_cached), eps_used=eps_used, step=step, merge_near=mthr, cache_hit=True))
            return list(levels_cached)

    levels: List[float] = []
    method = "dbscan"
    labels = None
    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps_used, min_samples=ms).fit(qarr).labels_
        for lab in set(labels) - {-1}:
            # Representative: median of quantized values for stability
            vals = qarr_flat[labels == lab]
            if vals.size == 0:
                continue
            # Filter by line_min_hits threshold (cluster size) to avoid tiny splits becoming lines when min_samples increases.
            if vals.size >= lm_hits:
                med = float(np.median(vals))
                levels.append(_q(med))
    except Exception:
        # Fallback: simple bucketing by eps_used
        method = "bucket"
        bins: Dict[int, List[float]] = {}
        for v in qarr_flat:
            key_b = round(v / eps_used) if eps_used > 0 else 0
            bins.setdefault(key_b, []).append(float(v))
        for _, vals in bins.items():
            if len(vals) >= lm_hits:
                # Use median on quantized values
                levels.append(float(np.median(vals)))

    # Already quantized; ensure unique and sorted
    levels_rounded = sorted({ _q(v) for v in levels })

    # Merge-near (default ≈ pip). Ensure threshold >= step to avoid grid jitter.
    mthr_final = float(max(mthr, step))
    merged: List[float] = []
    for v in levels_rounded:
        if not merged:
            merged.append(v); continue
        if abs(v - merged[-1]) <= mthr_final:
            merged[-1] = float((merged[-1] + v) / 2.0)
        else:
            merged.append(v)

    # Quality metrics
    if isinstance(quality_out, dict):
        quality_out.update(dict(
            n_prices=len(qarr),
            n_levels_clusters=len(levels),
            n_levels_raw=len(levels_rounded),
            n_levels_merged=len(merged),
            eps_used=eps_used,
            min_samples=ms,
            cluster_min_samples=ms,
            line_min_hits=lm_hits,
            method=method,
            step=step,
            merge_near=mthr,
            cache_hit=cache_hit,
        ))
        # silhouette (optional)
        try:
            if labels is not None and (set(labels) - {-1}):
                from sklearn.metrics import silhouette_score
                # filter noise (-1)
                mask = labels != -1
                if mask.sum() >= 2 and len(set(labels[mask])) >= 2:
                    quality_out["silhouette"] = float(silhouette_score(qarr[mask], labels[mask]))
        except Exception:
            pass

    # Save to cache
    if cache:
        try:
            horizontal_levels._cache_store[key] = tuple(merged)
            # bound cache size
            if len(horizontal_levels._cache_store) > 256:
                horizontal_levels._cache_store.pop(next(iter(horizontal_levels._cache_store)))
        except Exception:
            pass

    return merged

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
