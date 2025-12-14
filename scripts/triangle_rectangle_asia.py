from __future__ import annotations

import os
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.core.ta import add_atr_if_missing  # noqa: E402


def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    # Robust: accept tz-aware DatetimeIndex as-is
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["time", "timestamp", "datetime", "date"]:
            if col in df.columns:
                out = df.copy()
                out.index = pd.to_datetime(out[col], utc=True, errors="coerce")
                return out
        out = df.copy()
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        return out
    return df


def _linreg(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
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
    return float(m), float(b), float(s)


def _debug(msg: str) -> None:
    if os.getenv("PATTERN_DEBUG", "0") not in ("0", "", None):
        print(f"[PATTERN_DEBUG] {msg}")


def _apply_overrides(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base)
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            out[k] = v
    return out


def detect_triangle(
    df: pd.DataFrame,
    *,
    cfg: Dict[str, Any],
    session_override: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Minimal triangle detector:
    - Use last lookback bars
    - For window L in [min_bars, max_bars], fit upper (rolling max) and lower (rolling min) lines via regression
    - Require slopes with opposite signs and convergence near window end
    - Emit on breakout of next bar beyond the two lines
    Returns list of dict(time, index, side, entry, delta, quality, kind)
    """
    if df.empty:
        return []

    cfg = _apply_overrides(cfg, session_override)
    lookback = int(cfg.get("lookback", 600))
    Lmin = int(cfg.get("min_bars", 10))
    Lmax = int(cfg.get("max_bars", 60))
    converge_ratio_max = float(cfg.get("converge_ratio_max", 0.4))
    forbid_same_sign = bool(cfg.get("slope_same_sign_forbid", True))
    breakout_sigma_k = float(cfg.get("breakout_sigma_k", 1.0))

    df2 = df.copy()
    df2 = add_atr_if_missing(df2)
    df2 = _ensure_time_index(df2)

    highs = pd.to_numeric(df2["high"], errors="coerce").astype(float).values
    lows = pd.to_numeric(df2["low"], errors="coerce").astype(float).values
    closes = pd.to_numeric(df2["close"], errors="coerce").astype(float).values
    atr = pd.to_numeric(df2.get("atr"), errors="coerce").astype(float).values

    n = len(df2)
    start = max(0, n - int(lookback))
    idx = np.arange(n)

    signals: List[Dict[str, Any]] = []

    for b in range(start + Lmin, n - 1):  # b is window end (exclusive)
        # Try a single representative L to keep it light, plus one more
        for L in (Lmin, min(Lmax, Lmin + 10)):
            a = b - L
            if a < 2:
                continue
            sub_h = highs[a:b]
            sub_l = lows[a:b]
            sub_c = closes[a:b]
            x = np.arange(L, dtype=float)

            # Upper line: regress on local maxima (use highs)
            mu, cu, su = _linreg(x, sub_h)
            # Lower line: regress on local minima (use lows)
            ml, cl, sl = _linreg(x, sub_l)

            if forbid_same_sign and (mu == 0 or ml == 0 or (mu > 0 and ml > 0) or (mu < 0 and ml < 0)):
                continue

            # Convergence check: distance early vs late
            d0 = (mu * 0 + cu) - (ml * 0 + cl)  # at window start
            d1 = (mu * (L - 1) + cu) - (ml * (L - 1) + cl)  # at window end
            d0 = float(abs(d0))
            d1 = float(abs(d1))
            if d0 <= 1e-9:
                continue
            ratio = d1 / d0
            if not (0.0 <= ratio <= max(1.0, converge_ratio_max)):
                continue

            # Breakout check at next bar b
            mid_u = mu * L + cu
            mid_l = ml * L + cl
            band_u = mid_u + breakout_sigma_k * su
            band_l = mid_l - breakout_sigma_k * sl

            hi_b = highs[b]
            lo_b = lows[b]

            side = 0
            if np.isfinite(hi_b) and hi_b >= band_u:
                side = +1
                entry = closes[b]
            elif np.isfinite(lo_b) and lo_b <= band_l:
                side = -1
                entry = closes[b]
            else:
                continue

            delta = 0.5 * (atr[b] if np.isfinite(atr[b]) else np.nan)
            if not (np.isfinite(delta) and delta > 0):
                # fallback small delta
                delta = max(1e-6, 0.5 * np.nanmean(atr[max(0, a-20):b]))

            # Quality: tighter convergence and steeper opposition => higher
            opp = 1.0 - float(abs(abs(mu) - abs(ml)) / (abs(mu) + abs(ml) + 1e-9))
            conv = 1.0 - min(1.0, max(0.0, ratio))
            vol = float(np.nanmean(atr[a:b]))
            sband = float(su + sl + 1e-9)
            tight = float(vol / sband) if sband > 0 else 0.0
            q = 0.5 * opp + 0.3 * conv + 0.2 * math.tanh(tight / 6.0)
            q = float(max(0.0, min(1.0, q)))

            signals.append(
                {
                    "time": pd.Timestamp(df2.index[b]),
                    "index": int(b),
                    "side": int(side),
                    "entry": float(entry),
                    "delta": float(delta),
                    "quality": float(q),
                    "kind": "triangle",
                }
            )
            break  # break L loop once found

    _debug(f"triangle: found {len(signals)} candidates")
    return signals


def detect_rectangle(
    df: pd.DataFrame,
    *,
    cfg: Dict[str, Any],
    session_override: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Simple rectangle detector:
    - Window L in [min_bars, max_bars]
    - Range = max(high)-min(low) small vs ATR
    - Breakout at next bar beyond range upper/lower with buffer
    """
    if df.empty:
        return []
    cfg = _apply_overrides(cfg, session_override)
    lookback = int(cfg.get("lookback", 600))
    Lmin = int(cfg.get("min_bars", 10))
    Lmax = int(cfg.get("max_bars", 60))
    max_range_atr_mult = float(cfg.get("max_range_atr_mult", 2.0))
    min_touches = int(cfg.get("min_touches", 3))
    buffer = float(cfg.get("breakout_buffer_pips", 0.0))

    df2 = add_atr_if_missing(_ensure_time_index(df.copy()))
    highs = pd.to_numeric(df2["high"], errors="coerce").astype(float).values
    lows = pd.to_numeric(df2["low"], errors="coerce").astype(float).values
    closes = pd.to_numeric(df2["close"], errors="coerce").astype(float).values
    atr = pd.to_numeric(df2.get("atr"), errors="coerce").astype(float).values

    n = len(df2)
    start = max(0, n - int(lookback))
    idx = np.arange(n)

    out: List[Dict[str, Any]] = []
    for b in range(start + Lmin, n - 1):
        for L in (Lmin, min(Lmax, Lmin + 10)):
            a = b - L
            if a < 2:
                continue
            hh = float(np.nanmax(highs[a:b]))
            ll = float(np.nanmin(lows[a:b]))
            rng = hh - ll
            atr_ref = float(np.nanmean(atr[a:b]))
            if not (np.isfinite(atr_ref) and atr_ref > 0):
                continue
            if rng > max_range_atr_mult * atr_ref:
                continue
            # Count touches near edges
            near_up = np.sum(highs[a:b] >= (hh - 0.1 * atr_ref))
            near_dn = np.sum(lows[a:b] <= (ll + 0.1 * atr_ref))
            if (near_up < min_touches) or (near_dn < min_touches):
                continue
            # Breakout next bar
            hi_b = highs[b]
            lo_b = lows[b]
            side = 0
            if np.isfinite(hi_b) and hi_b >= (hh + buffer):
                side = +1
            elif np.isfinite(lo_b) and lo_b <= (ll - buffer):
                side = -1
            else:
                continue
            entry = float(closes[b])
            delta = 0.5 * (atr[b] if np.isfinite(atr[b]) else atr_ref)
            q = float(max(0.0, min(1.0, 1.0 - (rng / (max_range_atr_mult * atr_ref)))))
            out.append(
                {
                    "time": pd.Timestamp(df2.index[b]),
                    "index": int(b),
                    "side": int(side),
                    "entry": float(entry),
                    "delta": float(delta),
                    "quality": float(q),
                    "kind": "rectangle",
                }
            )
            break
    _debug(f"rectangle: found {len(out)} candidates")
    return out


def detect_asia_box(
    df: pd.DataFrame,
    *,
    cfg: Dict[str, Any],
    session_override: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Asia Box breakout (UTC clock used; JST 9-17 == UTC 0-8 by default):
    - For each UTC-day, take [start_hour_utc, end_hour_utc) window
    - Compute box high/low and its ATR-based validity
    - After window, within check_after_hours, emit breakout signals when price moves outside the box with buffer
    """
    if df.empty:
        return []
    cfg = _apply_overrides(cfg, session_override)
    h0 = int(cfg.get("start_hour_utc", 0))
    h1 = int(cfg.get("end_hour_utc", 9))
    min_mult = float(cfg.get("min_range_atr_mult", 0.6))
    max_mult = float(cfg.get("max_range_atr_mult", 3.0))
    buffer = float(cfg.get("breakout_buffer_pips", 0.0))
    after_h = int(cfg.get("check_after_hours", 8))

    df2 = _ensure_time_index(df.copy())
    # force UTC index
    try:
        if df2.index.tz is None:
            df2.index = pd.to_datetime(df2.index, utc=True)
        else:
            df2.index = pd.to_datetime(df2.index).tz_convert("UTC")
    except Exception:
        df2.index = pd.to_datetime(df2.index, utc=True)

    df2 = add_atr_if_missing(df2)
    highs = pd.to_numeric(df2["high"], errors="coerce").astype(float)
    lows = pd.to_numeric(df2["low"], errors="coerce").astype(float)
    closes = pd.to_numeric(df2["close"], errors="coerce").astype(float)
    atr = pd.to_numeric(df2.get("atr"), errors="coerce").astype(float)

    # Unique UTC days
    dates = pd.to_datetime(df2.index.date)
    unique_days = np.unique(dates)

    out: List[Dict[str, Any]] = []
    for day in unique_days:
        # Ensure pandas Timestamp in UTC
        try:
            day_ts = pd.Timestamp(day)
        except Exception:
            day_ts = pd.to_datetime(day)
        if getattr(day_ts, 'tzinfo', None) is None:
            day_ts = day_ts.tz_localize("UTC")
        else:
            day_ts = day_ts.tz_convert("UTC")
        # window [h0, h1)
        t0 = day_ts + pd.Timedelta(hours=h0)
        t1 = day_ts + pd.Timedelta(hours=h1)
        asia = df2.loc[(df2.index >= t0) & (df2.index < t1)]
        if asia.empty:
            continue
        ah = float(np.nanmax(asia["high"]))
        al = float(np.nanmin(asia["low"]))
        rng = ah - al
        atr_ref = float(np.nanmedian(atr.loc[asia.index]))
        if not (np.isfinite(atr_ref) and atr_ref > 0):
            continue
        if (rng < min_mult * atr_ref) or (rng > max_mult * atr_ref):
            continue
        # After window: check breakouts
        t_after_end = t1 + pd.Timedelta(hours=after_h)
        aft = df2.loc[(df2.index >= t1) & (df2.index < t_after_end)]
        if aft.empty:
            continue
        # first breakout bar only
        broke = False
        for ts, row in aft.iterrows():
            hi = float(row["high"])
            lo = float(row["low"])
            cl = float(row["close"])
            ar = float(row["atr"]) if np.isfinite(row.get("atr", np.nan)) else atr_ref
            if hi >= ah + buffer:
                side = +1
            elif lo <= al - buffer:
                side = -1
            else:
                continue
            delta = 0.5 * ar
            # quality: narrower box and early breakout rewarded
            narrow = 1.0 - min(1.0, float(rng / (max_mult * atr_ref)))
            early = 1.0 - min(1.0, float((ts - t1).total_seconds() / (after_h * 3600.0 + 1e-9)))
            q = 0.6 * narrow + 0.4 * early
            q = float(max(0.0, min(1.0, q)))
            # Robust index lookup for tz-aware DatetimeIndex
            try:
                loc = df2.index.get_loc(ts)
                if isinstance(loc, slice):
                    idx_val = int(loc.start or 0)
                elif isinstance(loc, (np.ndarray, list)):
                    idx_val = int(loc[0]) if len(loc) else 0
                else:
                    idx_val = int(loc)
            except Exception:
                idx_val = int(df2.index.searchsorted(ts))

            out.append(
                {
                    "time": pd.Timestamp(ts),
                    "index": idx_val,
                    "side": int(side),
                    "entry": float(cl),
                    "delta": float(delta),
                    "quality": float(q),
                    "kind": "asia_box",
                }
            )
            broke = True
            break
    _debug(f"asia_box: found {len(out)} candidates")
    return out


def run_detection(
    df: pd.DataFrame,
    *,
    pattern: str,
    cfg: Dict[str, Any],
    session_override: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    pat = str(pattern).lower().strip()
    if pat == "triangle":
        return detect_triangle(df, cfg=cfg, session_override=session_override)
    if pat == "rectangle":
        return detect_rectangle(df, cfg=cfg, session_override=session_override)
    if pat in ("asia", "asia_box", "asia-box", "asiabox"):
        return detect_asia_box(df, cfg=cfg, session_override=session_override)
    return []


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Minimal detectors for triangle/rectangle/asia_box")
    p.add_argument("--pattern", required=True, choices=["triangle", "rectangle", "asia_box"], help="pattern kind")
    p.add_argument("--data", default="data/USDJPY_15m.csv")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    df = _ensure_time_index(df)
    df = add_atr_if_missing(df)

    cfg_all = {}
    try:
        import yaml  # type: ignore
        y = Path("config/patterns.yml")
        if y.exists():
            cfg_all = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    cfg = cfg_all.get(args.pattern, {}) if isinstance(cfg_all, dict) else {}
    sigs = run_detection(df, pattern=args.pattern, cfg=cfg)
    print(f"pattern={args.pattern} candidates={len(sigs)}")
