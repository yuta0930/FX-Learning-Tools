import numpy as np
import pandas as pd
from typing import Optional

# =========================
# Flag / Pennant Detector (drop-in)
# =========================

def _atr(high, low, close, window=14):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    prev_close = np.roll(close, 1)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr[0] = high[0] - low[0]
    alpha = 2.0 / (window + 1.0)
    atr = np.empty_like(tr)
    atr[0] = tr[:window].mean() if len(tr) >= window else tr[0]
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    return atr

def _pivots(series, lb=2, ub=2):
    x = np.asarray(series, dtype=float)
    n = len(x)
    is_max = np.zeros(n, dtype=bool)
    is_min = np.zeros(n, dtype=bool)
    for i in range(lb, n-ub):
        window = x[i-lb:i+ub+1]
        if np.argmax(window) == lb and (window[lb] > window[:lb]).all() and (window[lb] > window[lb+1:]).all():
            is_max[i] = True
        if np.argmin(window) == lb and (window[lb] < window[:lb]).all() and (window[lb] < window[lb+1:]).all():
            is_min[i] = True
    return is_max, is_min

def _fit_line(xs, ys):
    if len(xs) < 2:
        return 0.0, 0.0, 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x0 = x - x.mean()
    s, b = np.polyfit(x0, y, 1)
    intercept = b - s * (-x.mean())
    yhat = s * x0 + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 1e-12 else 0.0
    return float(s), float(intercept), float(r2)

def _line_y(slope, intercept, x):
    return slope * x + intercept

def _normalized_slope(slope, price_scale):
    if price_scale <= 0:
        return 0.0
    return slope / price_scale

def detect_flags_pennants(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    time_col: Optional[str] = None,
    atr_window: int = 14,
    pole_max_bars: int = 20,
    pole_min_atr: float = 2.5,
    cons_min_bars: int = 8,
    cons_max_bars: int = 24,
    pivot_lb: int = 2,
    pivot_ub: int = 2,
    slope_abs_max_norm: float = 0.004,
    parallel_tol: float = 0.20,
    converge_tol: float = 0.25,
    width_max_atr: float = 3.0,
    contraction_min: float = 0.75,
    breakout_buffer_atr: float = 0.30,
    require_breakout: bool = False,
):
    if any(c not in df.columns for c in [high_col, low_col, close_col]):
        raise ValueError("DataFrame must have high/low/close columns")

    highs = df[high_col].to_numpy(dtype=float)
    lows  = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)
    n = len(df)
    if n < max(atr_window + pole_max_bars + cons_max_bars + 5, 60):
        return []

    atr = _atr(highs, lows, close, window=atr_window)
    is_h, _ = _pivots(highs, pivot_lb, pivot_ub)
    _, is_l = _pivots(lows,  pivot_lb, pivot_ub)

    patterns = []

    for i in range(atr_window + pole_max_bars + cons_min_bars, n):
        pole_end = i - cons_min_bars
        pole_start_candidates = range(max(atr_window, pole_end - pole_max_bars), pole_end)

        best_pole = None  # (start, end, dir, atr_units, score)
        for s in pole_start_candidates:
            price_move = close[pole_end] - close[s]
            atr_units = abs(price_move) / (atr[pole_end] + 1e-12)
            if atr_units >= pole_min_atr:
                d = "bull" if price_move > 0 else "bear"
                score = atr_units / max(1, pole_end - s)
                if best_pole is None or score > best_pole[-1]:
                    best_pole = (s, pole_end, d, atr_units, score)
        if best_pole is None:
            continue

        pole_s, pole_e, pole_dir, pole_atr_units, _ = best_pole

        cons_s = pole_e + 1
        cons_e = i
        cons_len = cons_e - cons_s + 1
        if cons_len < cons_min_bars or cons_len > cons_max_bars:
            continue

        cons_high = highs[cons_s:cons_e+1].max()
        cons_low  = lows[cons_s:cons_e+1].min()
        cons_width = cons_high - cons_low
        if cons_width > width_max_atr * atr[cons_e]:
            continue

        atr_cons = atr[cons_s:cons_e+1].mean()
        atr_pole = atr[max(pole_s-atr_window, 0):pole_e+1].mean()
        if atr_cons > contraction_min * atr_pole:
            continue

        idxs   = np.arange(cons_s, cons_e+1)
        hi_idx = idxs[is_h[cons_s:cons_e+1]]
        lo_idx = idxs[is_l[cons_s:cons_e+1]]

        if len(hi_idx) < 2 or len(lo_idx) < 2:
            k = min(4, len(idxs))
            top_hi_idx = idxs[np.argsort(highs[cons_s:cons_e+1])[-k:]] if k > 1 else np.array([], dtype=int)
            top_lo_idx = idxs[np.argsort(lows[cons_s:cons_e+1])[:k]]  if k > 1 else np.array([], dtype=int)
            hi_idx = np.sort(top_hi_idx)
            lo_idx = np.sort(top_lo_idx)

        uh_slope, uh_inter, uh_r2 = _fit_line(hi_idx, highs[hi_idx])
        lh_slope, lh_inter, lh_r2 = _fit_line(lo_idx, lows[lo_idx])

        price_scale = close[cons_s:cons_e+1].mean()
        uh_slope_n = _normalized_slope(uh_slope, price_scale)
        lh_slope_n = _normalized_slope(lh_slope, price_scale)

        if abs(uh_slope_n) > slope_abs_max_norm or abs(lh_slope_n) > slope_abs_max_norm:
            continue

        pattern_type = None
        if np.sign(uh_slope_n) == np.sign(lh_slope_n) and np.sign(uh_slope_n) != 0:
            rel_diff = abs(abs(uh_slope_n) - abs(lh_slope_n)) / max(abs(uh_slope_n), abs(lh_slope_n), 1e-9)
            if rel_diff <= parallel_tol:
                pattern_type = "flag"
        else:
            rel_mag = abs(abs(uh_slope_n) - abs(lh_slope_n)) / max(abs(uh_slope_n), abs(lh_slope_n), 1e-9)
            if rel_mag <= converge_tol:
                pattern_type = "pennant"

        if pattern_type is None:
            continue

        cons_ret = close[cons_e] - close[cons_s]
        drift_ok = (pole_dir == "bull" and cons_ret >= -0.25*atr_cons) or (pole_dir == "bear" and cons_ret <= 0.25*atr_cons)
        if not drift_ok:
            continue

        breakout_idx = None
        upper_y_end = _line_y(uh_slope, uh_inter, cons_e)
        lower_y_end = _line_y(lh_slope, lh_inter, cons_e)
        if pole_dir == "bull":
            trigger = upper_y_end + breakout_buffer_atr * atr[cons_e]
            if highs[cons_e] >= trigger:
                breakout_idx = cons_e
        else:
            trigger = lower_y_end - breakout_buffer_atr * atr[cons_e]
            if lows[cons_e] <= trigger:
                breakout_idx = cons_e

        q_pole = min(pole_atr_units / 5.0, 1.0)
        q_fit  = max(min((uh_r2 + lh_r2) / 2.0, 1.0), 0.0)
        sym = 1.0 - (abs(abs(uh_slope_n) - abs(lh_slope_n)) / max(abs(uh_slope_n), abs(lh_slope_n), 1e-9))
        contr_ratio = atr_cons / (atr_pole + 1e-12)
        q_contr = min(1.0, max(0.0, (1.0 - contr_ratio) / (1.0 - contraction_min + 1e-9)))
        quality = float(np.clip(0.35*q_pole + 0.35*q_fit + 0.2*sym + 0.1*q_contr, 0, 1))

        pole_height = abs(close[pole_e] - close[pole_s])
        if pole_dir == "bull":
            entry  = max(trigger, upper_y_end)
            stop   = lower_y_end - 0.25*atr[cons_e]
            target = entry + pole_height
        else:
            entry  = min(trigger, lower_y_end)
            stop   = upper_y_end + 0.25*atr[cons_e]
            target = entry - pole_height

        patterns.append({
            "kind": pattern_type,
            "dir": pole_dir,
            "start_idx": int(cons_s),
            "end_idx": int(cons_e),
            "pole_start_idx": int(pole_s),
            "pole_end_idx": int(pole_e),
            "breakout_idx": int(breakout_idx) if breakout_idx is not None else None,
            "upper_line": (float(uh_slope), float(uh_inter), float(uh_r2)),
            "lower_line": (float(lh_slope), float(lh_inter), float(lh_r2)),
            "quality_score": quality,
            "atr_cons": float(atr_cons),
            "atr_pole": float(atr_pole),
            "contraction_ratio": float(atr_cons / (atr_pole + 1e-12)),
            "entry": float(entry),
            "stop": float(stop),
            "target": float(target),
        })

    return patterns

def detect_flag_pennant(df, **kwargs) -> pd.DataFrame:
    """互換ラッパー：DataFrameで返す"""
    patterns = detect_flags_pennants(df, **kwargs)
    if not patterns:
        return pd.DataFrame(columns=[
            'kind','dir','start_idx','end_idx','pole_start_idx','pole_end_idx',
            'breakout_idx','entry','stop','target','quality_score','atr_cons','atr_pole','contraction_ratio',
            'upper_slope','upper_intercept','upper_r2','lower_slope','lower_intercept','lower_r2'
        ])
    rows = []
    for p in patterns:
        rows.append([
            p['kind'], p['dir'], p['start_idx'], p['end_idx'], p['pole_start_idx'], p['pole_end_idx'],
            p['breakout_idx'], p['entry'], p['stop'], p['target'], p['quality_score'], p['atr_cons'], p['atr_pole'], p['contraction_ratio'],
            p['upper_line'][0], p['upper_line'][1], p['upper_line'][2], p['lower_line'][0], p['lower_line'][1], p['lower_line'][2]
        ])
    cols = ['kind','dir','start_idx','end_idx','pole_start_idx','pole_end_idx',
            'breakout_idx','entry','stop','target','quality_score','atr_cons','atr_pole','contraction_ratio',
            'upper_slope','upper_intercept','upper_r2','lower_slope','lower_intercept','lower_r2']
    return pd.DataFrame(rows, columns=cols)
