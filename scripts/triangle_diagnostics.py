#!/usr/bin/env python
"""
Triangle pattern diagnostics script.
Instruments the detect_triangles logic (as implemented in app.py) to produce
frequency and filter drop-off statistics, plus quality distribution summary.
Outputs a JSON and a Markdown summary under reports/triangle_diag_YYYYMMDD.
"""
from __future__ import annotations
import json, math, argparse, statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Minimal reuse of logic from app.py (copy of detect_triangles core with instrumentation)

def _atr(high, low, close, window=14):
    high = np.asarray(high, dtype=float); low = np.asarray(low, dtype=float); close = np.asarray(close, dtype=float)
    prev_close = np.roll(close, 1)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr[0] = high[0] - low[0]
    alpha = 1.0 / max(1.0, float(window))
    atr = np.empty_like(tr)
    atr[0] = tr[:window].mean() if len(tr) >= window else tr[0]
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    return atr

def _pivots(series, lb=2, ub=2):
    x = np.asarray(series, dtype=float)
    n = len(x)
    is_max = np.zeros(n, dtype=bool); is_min = np.zeros(n, dtype=bool)
    for i in range(lb, n-ub):
        window = x[i-lb:i+ub+1]
        if np.argmax(window) == lb and (window[lb] > window[:lb]).all() and (window[lb] > window[lb+1:]).all():
            is_max[i] = True
        if np.argmin(window) == lb and (window[lb] < window[:lb]).all() and (window[lb] < window[lb+1:]).all():
            is_min[i] = True
    return is_max, is_min

def _fit_line(xs, ys):
    if len(xs) < 2: return 0.0, 0.0, 0.0
    x = np.asarray(xs, dtype=float); y = np.asarray(ys, dtype=float)
    x0 = x - x.mean(); s, b = np.polyfit(x0, y, 1)
    intercept = b - s * (-x.mean())
    yhat = s * x0 + b
    ss_res = np.sum((y - yhat)**2); ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 1e-12 else 0.0
    return float(s), float(intercept), float(r2)

def detect_triangles_instrumented(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    params = dict(
        high_col="high", low_col="low", close_col="close", atr_window=14,
        cons_min_bars=12, cons_max_bars=40, pivot_lb=2, pivot_ub=2,
    flat_tol_norm=0.0012, converge_min=0.20, width_max_atr=3.5,
    contraction_max_ratio=0.95,  # allow <= 95% of previous ATR by default (strict)
    r2_min=0.20, parallel_tol=0.22, breakout_buffer_atr=0.25,
        confirm_bars=1, pretrend_win=24, require_breakout=False, last_N=3000,
        e_step=2, len_step=2
    )
    params.update(kwargs)

    hc, lc, cc = params['high_col'], params['low_col'], params['close_col']
    for c in (hc, lc, cc):
        if c not in df.columns:
            raise ValueError("Missing OHLC column: " + c)
    if params['last_N'] is not None and len(df) > params['last_N']:
        df = df.iloc[-params['last_N']:]  # safe slice

    highs = df[hc].to_numpy(dtype=float); lows = df[lc].to_numpy(dtype=float); close = df[cc].to_numpy(dtype=float)
    n = len(df)
    if n < max(params['atr_window'] + params['cons_max_bars'] + 5, 80):
        return {"patterns": [], "stats": {"n_bars": n}}
    atr = _atr(highs, lows, close, window=params['atr_window'])
    is_h, _ = _pivots(highs, params['pivot_lb'], params['pivot_ub'])
    _, is_l = _pivots(lows,  params['pivot_lb'], params['pivot_ub'])

    counters = {"loop_e":0, "len_try":0, "fail_width_raw":0, "fail_atr_cons":0, "fail_pivots":0,
                "fail_r2":0, "fail_width_e":0, "fail_converge":0, "fail_type":0, "accepted":0}
    qualities: List[float] = []
    converge_list: List[float] = []
    width_e_list: List[float] = []
    patterns: List[Dict[str, Any]] = []

    atr_window = params['atr_window']; cons_min_bars = params['cons_min_bars']; cons_max_bars = params['cons_max_bars']; e_step=params['e_step']; len_step=params['len_step']
    min_pivot_points = 3

    def _pretrend_slope(end_idx, win: int) -> float:
        j1 = max(0, end_idx - cons_min_bars)
        j0 = max(0, j1 - win)
        if j1 - j0 < 5: return 0.0
        x = np.arange(j1-j0+1, dtype=float); y = close[j0:j1+1]
        xm, ym = x.mean(), y.mean(); den = ((x-xm)**2).sum()
        if den <= 0: return 0.0
        return float(((x-xm)*(y-ym)).sum()/den)

    def _confirm_break(e_idx: int, dir_side: str, m_up, b_up, m_lo, b_lo):
        seq = 0; j = e_idx
        while j < n:
            up_y = m_up*j + b_up; lo_y = m_lo*j + b_lo
            thr = params['breakout_buffer_atr'] * atr[j]
            c = close[j]
            ok = (c >= up_y + thr) if dir_side=="up" else (c <= lo_y - thr)
            seq = seq + 1 if ok else 0
            if seq >= max(1, int(params['confirm_bars'])):
                if dir_side == "up":
                    entry = max(up_y + thr, c); stop = lo_y - 0.25*atr[j]
                else:
                    entry = min(lo_y - thr, c); stop = up_y + 0.25*atr[j]
                return True, j, float(entry), float(stop)
            if j - e_idx > 3: break
            j += 1
        return False, None, np.nan, np.nan

    for e in range(atr_window + cons_min_bars, n, e_step):
        counters['loop_e'] += 1
        best = None
        for cons_len in range(cons_min_bars, cons_max_bars+1, len_step):
            counters['len_try'] += 1
            s = e - cons_len + 1
            if s < atr_window:
                continue
            width_raw = highs[s:e+1].max() - lows[s:e+1].min()
            if width_raw > params['width_max_atr'] * atr[e] * 1.5:
                counters['fail_width_raw'] += 1; continue
            atr_cons = atr[s:e+1].mean(); atr_prev = atr[max(s-14,0):s].mean()
            # Require some contraction vs previous volatility, but configurable
            if atr_cons > atr_prev * params['contraction_max_ratio']:
                counters['fail_atr_cons'] += 1; continue
            idxs = np.arange(s, e+1)
            hi_idx = idxs[is_h[s:e+1]]; lo_idx = idxs[is_l[s:e+1]]
            if len(hi_idx) < min_pivot_points or len(lo_idx) < min_pivot_points:
                k = min(6, len(idxs))
                if k < min_pivot_points:
                    counters['fail_pivots'] += 1; continue
                top_hi_idx = idxs[np.argsort(highs[s:e+1])[-k:]]
                top_lo_idx = idxs[np.argsort(lows[s:e+1])[:k]]
                hi_idx = np.sort(top_hi_idx[:max(min_pivot_points, len(top_hi_idx)//2)])
                lo_idx = np.sort(top_lo_idx[:max(min_pivot_points, len(top_lo_idx)//2)])
            if len(hi_idx) < min_pivot_points or len(lo_idx) < min_pivot_points:
                counters['fail_pivots'] += 1; continue
            uh_slope, uh_inter, uh_r2 = _fit_line(hi_idx, highs[hi_idx]); lh_slope, lh_inter, lh_r2 = _fit_line(lo_idx, lows[lo_idx])
            price_scale = close[s:e+1].mean();
            def _norm_slope(sl): return 0.0 if price_scale <= 0 else sl/price_scale
            uh_n = _norm_slope(uh_slope); lh_n = _norm_slope(lh_slope)
            if (uh_r2 + lh_r2)/2.0 < params['r2_min']:
                counters['fail_r2'] += 1; continue
            width_s = (uh_slope*s + uh_inter) - (lh_slope*s + lh_inter)
            width_e = (uh_slope*e + uh_inter) - (lh_slope*e + lh_inter)
            if width_s <= 0 or width_e <= 0:
                counters['fail_width_e'] += 1; continue
            if width_e > params['width_max_atr'] * atr[e]:
                counters['fail_width_e'] += 1; continue
            converge = (width_s - width_e) / max(width_s, 1e-9)
            if converge < params['converge_min']:
                counters['fail_converge'] += 1; continue
            tri_type = None
            if abs(uh_n) <= params['flat_tol_norm'] and lh_n > params['flat_tol_norm']:
                tri_type = 'ascending_triangle'
            elif abs(lh_n) <= params['flat_tol_norm'] and uh_n < -params['flat_tol_norm']:
                tri_type = 'descending_triangle'
            else:
                if np.sign(uh_n) != np.sign(lh_n) and np.sign(uh_n)!=0 and np.sign(lh_n)!=0:
                    rel = abs(abs(uh_n) - abs(lh_n)) / max(abs(uh_n), abs(lh_n), 1e-9)
                    if rel <= params['parallel_tol']:
                        tri_type = 'sym_triangle'
            if tri_type is None:
                counters['fail_type'] += 1; continue
            pre_slope = 0.0  # omit pretrend for diagnostics speed
            expect = 'up' if tri_type=='ascending_triangle' else ('down' if tri_type=='descending_triangle' else ('up' if close[e] >= ( (uh_slope*e+uh_inter)+(lh_slope*e+lh_inter) )*0.5 else 'down'))
            broken, b_idx, entry_cand, stop_cand = _confirm_break(e, expect, uh_slope, uh_inter, lh_slope, lh_inter)
            height = width_s
            if not broken:
                up_e = uh_slope*e + uh_inter; lo_e = lh_slope*e + lh_inter
                entry = up_e if expect=='up' else lo_e
                stop = lo_e - 0.25*atr[e] if expect=='up' else up_e + 0.25*atr[e]
                target = entry + height if expect=='up' else entry - height
            else:
                entry = entry_cand; stop = stop_cand; target = entry + height if expect=='up' else entry - height
            fit_q = max(0.0, min(1.0, (uh_r2 + lh_r2)/2.0))
            conv_q = max(0.0, min(1.0, (converge - params['converge_min']) / max(1e-9, 1.0 - params['converge_min'])))
            flat_thr = params['flat_tol_norm']
            flat_q = 1.0 - min(1.0, abs(uh_n)/flat_thr) if tri_type=='ascending_triangle' else (1.0 - min(1.0, abs(lh_n)/flat_thr) if tri_type=='descending_triangle' else 1.0 - min(1.0, abs(abs(uh_n)-abs(lh_n))/max(abs(uh_n),abs(lh_n),1e-9)))
            touch_score = min(len(hi_idx), len(lo_idx)); touch_q = min(1.0, (touch_score-2)/3.0)
            pre_q = 0.0 if pre_slope==0 else (1.0 if (tri_type=='ascending_triangle' and pre_slope>0) or (tri_type=='descending_triangle' and pre_slope<0) else 0.5)
            quality = float(np.clip(0.25*fit_q + 0.35*conv_q + 0.15*flat_q + 0.15*touch_q + 0.10*pre_q, 0, 1))
            counters['accepted'] += 1
            qualities.append(quality); converge_list.append(converge); width_e_list.append(width_e)
            patterns.append({"type":tri_type, "dir":"bull" if expect=='up' else 'bear', "start_idx":int(s), "end_idx":int(e), "quality":quality, "converge":converge, "width_e":width_e})
    stats = counters
    if qualities:
        stats.update({
            "quality_mean": statistics.mean(qualities),
            "quality_median": statistics.median(qualities),
            "quality_p80": float(np.quantile(qualities, 0.80)),
            "quality_p95": float(np.quantile(qualities, 0.95)),
            "n_patterns": len(patterns)
        })
    stats.update({"converge_mean": statistics.mean(converge_list) if converge_list else None,
                  "width_e_mean": statistics.mean(width_e_list) if width_e_list else None,
                  "n_bars": n})
    return {"patterns": patterns, "stats": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/USDJPY_15m.csv')
    ap.add_argument('--last-N', type=int, default=3000)
    ap.add_argument('--converge-min', type=float, default=0.20)
    ap.add_argument('--r2-min', type=float, default=0.20)
    ap.add_argument('--width-max-atr', type=float, default=3.5)
    ap.add_argument('--contraction-max-ratio', type=float, default=1.10, help='Allow ATR_cons <= ATR_prev * ratio (>=1.0 relaxes)')
    ap.add_argument('--quality-min', type=float, default=0.50)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    # Try to parse time index
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df.set_index('time', inplace=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.set_index('timestamp', inplace=True)
    else:
        df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.sort_index()

    res = detect_triangles_instrumented(df,
        last_N=args.last_N,
        converge_min=args.converge_min,
        r2_min=args.r2_min,
        width_max_atr=args.width_max_atr,
        contraction_max_ratio=args.contraction_max_ratio,
    )

    patterns = res['patterns']; stats = res['stats']
    high_q = [p for p in patterns if p['quality'] >= args.quality_min]
    low_hint = [p for p in patterns if 0.40 <= p['quality'] < args.quality_min]
    stats.update({
        'high_quality_count': len(high_q),
        'low_quality_hint_count': len(low_hint),
    })

    out_root = Path(args.out_dir) if args.out_dir else Path('reports') / f"triangle_diag_{datetime.now().strftime('%Y%m%d')}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'summary.json').write_text(json.dumps({'stats': stats, 'sample_patterns': patterns[:15]}, ensure_ascii=False, indent=2), encoding='utf-8')
    # Simple markdown for quick view
    md_lines = ["# Triangle Diagnostics", "", "## Stats", json.dumps(stats, ensure_ascii=False, indent=2)]
    (out_root / 'summary.md').write_text('\n'.join(md_lines), encoding='utf-8')
    print("Written:", out_root / 'summary.json')
    print("High-quality patterns:", stats.get('high_quality_count'))
    print("Low-quality hints:", stats.get('low_quality_hint_count'))
    print("Total accepted:", stats.get('accepted'))

if __name__ == '__main__':
    main()
