from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class FlagPennantSignal:
    t0: pd.Timestamp
    index: int
    side: int  # +1 long, -1 short (direction of the pole)
    entry: float
    delta: float  # price distance for triple-barrier
    quality: float
    kind: str  # "flag" or "pennant"
    # Additional diagnostics for filters/overrides
    slope_abs_atr: float | None = None
    contraction_pct: float | None = None


def _linreg(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return slope, intercept, and residual std of a simple linear regression."""
    if len(x) < 2:
        return 0.0, float(y[-1] if len(y) else 0.0), 0.0
    x = x.astype(float)
    y = y.astype(float)
    xm = x.mean()
    ym = y.mean()
    xx = ((x - xm) * (x - xm)).sum()
    if xx <= 0:
        return 0.0, float(ym), float(np.std(y))
    m = ((x - xm) * (y - ym)).sum() / xx
    b = ym - m * xm
    resid = y - (m * x + b)
    s = float(np.std(resid))
    return float(m), float(b), s


def detect_flag_pennant_simplified(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    lookback: int = 220,
    n_push: int = 30,
    min_flag_bars: int = 8,
    max_flag_bars: int = 40,
    sigma_k: float = 1.0,
    pole_min_atr: float = 2.0,
    # New thresholds
    flag_slope_max_atr: float = 0.15,
    contraction_percentile: float = 0.2,
) -> List[FlagPennantSignal]:
    """
    Minimal, import-safe Flag/Pennant detector approximating the app's logic:
    - Detect an impulsive pole over last n_push bars with |Δclose|/ATR >= pole_min_atr
    - Then find a consolidation window [min_flag_bars, max_flag_bars] with shrinking volatility
      and near-linear drift (pennant if slope≈0, flag if slope opposes pole)
    - Emit signal on breakout beyond regression band (mid ± sigma_k*resid_std) in the pole direction

    Returns a list of FlagPennantSignal for breakout bars within the last `lookback`.
    """
    required = {close_col, high_col, low_col, atr_col}
    if not required.issubset(df.columns):
        return []
    if df.empty:
        return []

    closes = pd.to_numeric(df[close_col], errors="coerce").astype(float).values
    highs = pd.to_numeric(df[high_col], errors="coerce").astype(float).values
    lows = pd.to_numeric(df[low_col], errors="coerce").astype(float).values
    atr = pd.to_numeric(df[atr_col], errors="coerce").astype(float).values

    n = len(df)
    start = max(0, n - lookback)
    idx = np.arange(n)
    out: List[FlagPennantSignal] = []

    # Precompute rolling window ranges (high-low) for each candidate L.
    # The previous implementation built per-candidate historical distributions with nested loops,
    # which is too slow for tests and small evaluations.
    hs = pd.Series(highs)
    ls = pd.Series(lows)
    range_by_L: dict[int, np.ndarray] = {}
    q_by_L: dict[int, float] = {}
    for L in range(min_flag_bars, max_flag_bars + 1):
        roll_hi = hs.rolling(L, min_periods=L).max().to_numpy()
        roll_lo = ls.rolling(L, min_periods=L).min().to_numpy()
        rng = (roll_hi - roll_lo).astype(float)
        range_by_L[L] = rng
        try:
            q_by_L[L] = float(np.nanquantile(rng, max(0.0, min(1.0, contraction_percentile))))
        except Exception:
            q_by_L[L] = float("nan")

    # Scan for poles then consolidation + breakout
    for i in range(start + n_push + min_flag_bars, n):
        # Pole ending at pivot 'a' just before consolidation starts
        a = i - min_flag_bars
        if a - n_push < 1:
            continue
        d_close = closes[a] - closes[a - n_push]
        atr_a = atr[a]
        if not np.isfinite(atr_a) or atr_a <= 1e-12:
            continue
        pole_atr = abs(d_close) / atr_a
        if pole_atr < pole_min_atr:
            continue
        pole_side = 1 if d_close > 0 else -1

        # Try different consolidation lengths
        found_breakout = False
        best_sig: Optional[FlagPennantSignal] = None
        for L in range(min_flag_bars, max_flag_bars + 1):
            b = a + L  # end of consolidation window
            if b >= n:
                break

            win_x = idx[a:b] - idx[a]
            win_y = closes[a:b]
            m, c, s = _linreg(win_x.astype(float), win_y.astype(float))
            # Consolidation: residual sigma small vs ATR and slope not too large
            if not (np.isfinite(s) and np.isfinite(m)):
                continue
            ref_slice = atr[max(0, a - n_push):a]
            ref_mean = float(np.nanmean(ref_slice)) if ref_slice.size > 0 else float(atr_a)
            if not np.isfinite(ref_mean) or ref_mean <= 1e-12:
                ref_mean = float(atr_a)
            # Slope normalization and contraction checks
            slope_abs_atr = abs(float(m)) / max(ref_mean, 1e-12)
            slope_ok = slope_abs_atr <= max(0.0, float(flag_slope_max_atr))

            # Window range vs historical distribution percentile as contraction
            end_idx = b - 1
            win_range = float(range_by_L.get(L, np.asarray([np.nan]))[end_idx]) if end_idx >= 0 else 0.0
            q = float(q_by_L.get(L, float("nan")))
            if not np.isfinite(q) or q <= 1e-12:
                q = win_range
            contraction_ok = win_range <= max(q, 1e-12)

            # Residual sigma guard (remain as a soft check)
            if s > 1.2 * ref_mean:
                continue
            # Slope must be not too aligned with the pole (prefer pullback/flat)
            if not slope_ok:
                continue

            # Check breakout at b (first bar after window end)
            mid_b = m * (b - a) + c
            upper_b = mid_b + sigma_k * s
            lower_b = mid_b - sigma_k * s

            if pole_side > 0:
                brk = highs[b] >= upper_b
            else:
                brk = lows[b] <= lower_b

            if brk:
                # Quality: emphasize stronger pole + tighter consolidation,
                # reward tighter-than-reference contraction and penalize steep slopes
                tight = atr_a / (s + 1e-9)
                base = 0.6 * (pole_atr / max(pole_min_atr, 1e-9)) + 0.4 * math.tanh(tight / 8.0)
                base = max(0.0, min(1.0, base))
                contr = 0.0 if (q <= 1e-12) else (win_range / q)
                contr_factor = max(0.0, min(1.0, 1.2 - float(contr)))
                slope_factor = 1.0 - min(1.0, float(slope_abs_atr) / max(1e-6, float(flag_slope_max_atr)))
                slope_factor = max(0.0, slope_factor)
                quality = base * (0.5 + 0.5 * contr_factor) * (0.5 + 0.5 * slope_factor)
                quality = float(max(0.0, min(1.0, quality)))

                delta = 0.5 * atr[b]  # default delta for labeling; script can override
                best_sig = FlagPennantSignal(
                    t0=pd.Timestamp(df.index[b]),
                    index=int(b),
                    side=int(pole_side),
                    entry=float(closes[b]),
                    delta=float(delta),
                    quality=quality,
                    kind="pennant" if abs(m) < 1e-4 else ("flag"),
                    slope_abs_atr=float(slope_abs_atr),
                    contraction_pct=float(0.0 if q <= 1e-12 else (win_range / q)),
                )
                found_breakout = True
                break
        if found_breakout and best_sig is not None:
            out.append(best_sig)

    return out
