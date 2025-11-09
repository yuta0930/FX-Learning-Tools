import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from joblib import Parallel, delayed
try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

# ---- Minimal utilities (copied from app.py to avoid Streamlit import side-effects) ----
def _atr(high, low, close, window=14):
    high = np.asarray(high, dtype=float)
    low  = np.asarray(low,  dtype=float)
    close= np.asarray(close,dtype=float)
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
    is_max = np.zeros(n, dtype=bool)
    is_min = np.zeros(n, dtype=bool)
    for i in range(lb, n-ub):
        window = x[i-lb:i+ub+1]
        if np.argmax(window) == lb and (window[lb] > window[:lb]).all() and (window[lb] > window[lb+1:]).all():
            is_max[i] = True
        if np.argmin(window) == lb and (window[lb] < window[:lb]).all() and (window[lb] < window[lb+1:]).all():
            is_min[i] = True
    return is_max, is_min

def _fit_line(xs, ys) -> Tuple[float,float,float]:
    if len(xs) < 2: return 0.0, 0.0, 0.0
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

def _norm_slope(slope, price_scale):
    return 0.0 if price_scale <= 0 else slope / float(price_scale)

# ---- Optional Numba accelerations ----
if HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _fit_line_numba(xs: np.ndarray, ys: np.ndarray) -> tuple:
        n = xs.size
        if n < 2:
            return 0.0, 0.0, 0.0
        # cast to float64
        x = xs.astype(np.float64)
        y = ys.astype(np.float64)
        # means
        x_sum = 0.0
        y_sum = 0.0
        for i in range(n):
            x_sum += x[i]
            y_sum += y[i]
        x_mean = x_sum / n
        y_mean = y_sum / n
        # slope (cov/var)
        num = 0.0
        den = 0.0
        for i in range(n):
            dx = x[i] - x_mean
            dy = y[i] - y_mean
            num += dx * dy
            den += dx * dx
        s = 0.0 if den <= 1e-18 else num / den
        b = y_mean - s * x_mean
        # R^2
        ss_res = 0.0
        ss_tot = 0.0
        for i in range(n):
            yhat = s * x[i] + b
            dres = y[i] - yhat
            ss_res += dres * dres
            dtot = y[i] - y_mean
            ss_tot += dtot * dtot
        r2 = 0.0 if ss_tot <= 1e-12 else (1.0 - ss_res / ss_tot)
        return float(s), float(b), float(r2)

    @njit(cache=True, fastmath=True)
    def _line_y_numba(slope: float, intercept: float, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float64)
        for i in range(x.size):
            out[i] = slope * x[i] + intercept
        return out

# ---- Optimized rectangle detector (kept in sync with app.py) ----
def detect_rectangles(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_window: int = 14,
    win_min_bars: int = 10,
    win_max_bars: int = 48,
    pivot_lb: int = 2,
    pivot_ub: int = 2,
    flat_tol_norm: float = 0.0009,
    drift_tol_norm: float = 0.0007,
    width_max_atr: float = 3.5,
    width_stability_max: float = 0.28,
    min_touches_each: int = 3,
    touch_tol_atr: float = 0.25,
    r2_min: float = 0.12,
    breakout_buffer_atr: float = 0.30,
    confirm_bars: int = 1,
    require_breakout: bool = False,
    last_N: Optional[int] = 2000,
    e_step: int = 3,
    len_step: int = 2,
    enable_low_pivot_fallback: bool = True,
    use_parallel: bool = False,
    n_jobs: int = -1,
    parallel_backend: str = "threads",
    parallel_batch_size: int = 16,
    use_numba: bool = False,
) -> List[Dict]:
    if any(c not in df.columns for c in [high_col, low_col, close_col]):
        raise ValueError("DataFrame must have high/low/close columns")
    if last_N is not None and len(df) > last_N:
        df = df.iloc[-last_N:].copy()

    highs = df[high_col].to_numpy(dtype=float)
    lows  = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)
    n = len(df)
    if n < max(atr_window + win_max_bars + 5, 80):
        return []

    atr = _atr(highs, lows, close, window=atr_window)
    is_h, _ = _pivots(highs, pivot_lb, pivot_ub)
    _, is_l = _pivots(lows,  pivot_lb, pivot_ub)

    H_cum = np.concatenate(([0], np.cumsum(is_h.astype(np.int32))))
    L_cum = np.concatenate(([0], np.cumsum(is_l.astype(np.int32))))
    close_cum = np.concatenate(([0.0], np.cumsum(close)))

    patterns: List[Dict] = []

    def _confirm_break(e_idx: int, side: str, m_up, b_up, m_lo, b_lo):
        seq = 0
        j = e_idx
        while j < n:
            up = _line_y(m_up, b_up, j)
            lo = _line_y(m_lo, b_lo, j)
            thr = breakout_buffer_atr * atr[j]
            c = close[j]
            ok = (c >= up + thr) if side == "up" else (c <= lo - thr)
            seq = seq + 1 if ok else 0
            if seq >= max(1,int(confirm_bars)):
                entry = max(up+thr, c) if side=="up" else min(lo-thr, c)
                stop  = lo - 0.25*atr[j] if side=="up" else up + 0.25*atr[j]
                return True, j, float(entry), float(stop)
            if j - e_idx > 3: break
            j += 1
        return False, None, np.nan, np.nan

    def _eval_e(e: int):
        best = None
        for L in range(win_min_bars, win_max_bars+1, len_step):
            s = e - L + 1
            if s < atr_window:
                continue
            h_cnt = int(H_cum[e+1] - H_cum[s])
            l_cnt = int(L_cum[e+1] - L_cum[s])
            idx_range = np.arange(s, e+1)
            hi_idx = idx_range[is_h[s:e+1]]
            lo_idx = idx_range[is_l[s:e+1]]
            if (h_cnt < 2 or l_cnt < 2):
                if not enable_low_pivot_fallback:
                    continue
                k = min(4, len(idx_range))
                if k < 2:
                    continue
                sub_high = highs[s:e+1]
                part_hi = np.argpartition(sub_high, -k)[-k:]
                sub_low = lows[s:e+1]
                part_lo = np.argpartition(sub_low, k-1)[:k]
                hi_idx = np.sort(idx_range[part_hi])[:max(2, len(part_hi)//2)]
                lo_idx = np.sort(idx_range[part_lo])[:max(2, len(part_lo)//2)]
                if len(hi_idx) < 2 or len(lo_idx) < 2:
                    continue
            if use_numba and HAVE_NUMBA:
                # cast indices to float for numba kernel
                uh_s, uh_b, uh_r2 = _fit_line_numba(hi_idx.astype(np.float64), highs[hi_idx].astype(np.float64))
                lh_s, lh_b, lh_r2 = _fit_line_numba(lo_idx.astype(np.float64), lows[lo_idx].astype(np.float64))
            else:
                uh_s, uh_b, uh_r2 = _fit_line(hi_idx, highs[hi_idx])
                lh_s, lh_b, lh_r2 = _fit_line(lo_idx, lows[lo_idx])
            price_scale = float((close_cum[e+1] - close_cum[s]) / L)
            uh_n = _norm_slope(uh_s, price_scale)
            lh_n = _norm_slope(lh_s, price_scale)
            mid_n = _norm_slope((uh_s + lh_s)/2.0, price_scale)
            if abs(uh_n) > flat_tol_norm or abs(lh_n) > flat_tol_norm: continue
            if (uh_r2 + lh_r2)/2.0 < r2_min: continue
            if abs(mid_n) > drift_tol_norm: continue
            dm = (uh_s - lh_s)
            db = (uh_b - lh_b)
            w_s = dm * s + db
            w_e = dm * e + db
            if min(w_s, w_e) <= 0: continue
            mean_i = 0.5 * (s + e)
            var_i = (L*L - 1) / 12.0
            w_mean = float(dm * mean_i + db)
            w_std  = float(abs(dm) * np.sqrt(var_i))
            if w_mean > width_max_atr * atr[e]: continue
            w_stab = (w_std / max(w_mean, 1e-9))
            if w_stab > width_stability_max: continue
            tol = touch_tol_atr * atr[e]
            if use_numba and HAVE_NUMBA:
                up_vals = _line_y_numba(uh_s, uh_b, hi_idx.astype(np.float64))
                lo_vals = _line_y_numba(lh_s, lh_b, lo_idx.astype(np.float64))
            else:
                up_vals = _line_y(uh_s, uh_b, hi_idx)
                lo_vals = _line_y(lh_s, lh_b, lo_idx)
            touch_up = int(np.sum(np.abs(highs[hi_idx] - up_vals) <= tol))
            touch_lo = int(np.sum(np.abs(lows[lo_idx]  - lo_vals) <= tol))
            if touch_up < min_touches_each or touch_lo < min_touches_each: continue
            up_e = float(uh_s * e + uh_b)
            lo_e = float(lh_s * e + lh_b)
            mid_e = (up_e + lo_e) * 0.5
            expect = "up" if close[e] >= mid_e else "down"
            broken, b_idx, entry, stop = _confirm_break(e, expect, uh_s, uh_b, lh_s, lh_b)
            if require_breakout and not broken: continue
            height = float(w_s)
            if np.isnan(entry):
                entry = up_e if expect=="up" else lo_e
                stop  = lo_e - 0.25*atr[e] if expect=="up" else up_e + 0.25*atr[e]
            target = entry + height if expect=="up" else entry - height
            fit_q   = max(0.0, min(1.0, (uh_r2 + lh_r2)/2.0))
            flat_q  = 1.0 - min(1.0, max(abs(uh_n), abs(lh_n)) / flat_tol_norm)
            stab_q  = 1.0 - min(1.0, w_stab / width_stability_max)
            touch_q = min(1.0, 0.5*min(1.0, touch_up/min_touches_each) + 0.5*min(1.0, touch_lo/min_touches_each))
            drift_q = 1.0 - min(1.0, abs(mid_n)/drift_tol_norm)
            quality = float(np.clip(0.30*fit_q + 0.25*flat_q + 0.20*stab_q + 0.15*touch_q + 0.10*drift_q, 0, 1))
            pat = {"type":"rectangle","dir":"bull" if expect=="up" else "bear","start_idx":int(s),"end_idx":int(e),"breakout_idx":int(b_idx) if b_idx is not None else None,
                   "upper_line":(float(uh_s),float(uh_b),float(uh_r2)),"lower_line":(float(lh_s),float(lh_b),float(lh_r2)),
                   "width_mean":w_mean,"width_std":w_std,"width_stability":w_stab,"touches_upper":int(touch_up),"touches_lower":int(touch_lo),
                   "quality_score":quality,"entry":float(entry),"stop":float(stop),"target":float(target)}
            cand = (quality, pat)
            if best is None or cand[0] > best[0]: best = cand
        return best[1] if best is not None else None

    e_list = list(range(atr_window + win_min_bars, n, e_step))
    if use_parallel and len(e_list) >= max(32, (n_jobs if isinstance(n_jobs,int) and n_jobs>0 else 4)*4):
        res = Parallel(
            n_jobs=n_jobs,
            prefer=("processes" if parallel_backend=="loky" else "threads"),
            batch_size=int(max(1, parallel_batch_size))
        )(delayed(_eval_e)(e) for e in e_list)
        patterns.extend([p for p in res if p is not None])
    else:
        for e in e_list:
            r = _eval_e(e)
            if r is not None:
                patterns.append(r)

    return patterns

def detect_rectangles_baseline(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_window: int = 14,
    win_min_bars: int = 10,
    win_max_bars: int = 48,
    pivot_lb: int = 2,
    pivot_ub: int = 2,
    flat_tol_norm: float = 0.0009,
    drift_tol_norm: float = 0.0007,
    width_max_atr: float = 3.5,
    width_stability_max: float = 0.28,
    min_touches_each: int = 3,
    touch_tol_atr: float = 0.25,
    r2_min: float = 0.12,
    breakout_buffer_atr: float = 0.30,
    confirm_bars: int = 1,
    require_breakout: bool = False,
    last_N: Optional[int] = 3000,
    e_step: int = 1,
    len_step: int = 1,
) -> List[Dict]:
    """旧アルゴリズム（最適化前）: argsort による極値補完 + 幅配列生成。比較用。
    注意: 速度測定専用であり品質要件が異なる可能性あり。
    """
    if any(c not in df.columns for c in [high_col, low_col, close_col]):
        raise ValueError("DataFrame must have high/low/close columns")
    if last_N is not None and len(df) > last_N:
        df = df.iloc[-last_N:].copy()
    highs = df[high_col].to_numpy(dtype=float)
    lows  = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)
    n = len(df)
    if n < max(atr_window + win_max_bars + 5, 80):
        return []
    atr = _atr(highs, lows, close, window=atr_window)
    is_h, _ = _pivots(highs, pivot_lb, pivot_ub)
    _, is_l = _pivots(lows,  pivot_lb, pivot_ub)
    patterns: List[Dict] = []

    def _confirm_break(e_idx: int, side: str, m_up, b_up, m_lo, b_lo):
        seq = 0
        j = e_idx
        while j < n:
            up = _line_y(m_up, b_up, j)
            lo = _line_y(m_lo, b_lo, j)
            thr = breakout_buffer_atr * atr[j]
            c = close[j]
            ok = (c >= up + thr) if side == "up" else (c <= lo - thr)
            seq = seq + 1 if ok else 0
            if seq >= max(1,int(confirm_bars)):
                entry = max(up+thr, c) if side=="up" else min(lo-thr, c)
                stop  = lo - 0.25*atr[j] if side=="up" else up + 0.25*atr[j]
                return True, j, float(entry), float(stop)
            if j - e_idx > 3: break
            j += 1
        return False, None, np.nan, np.nan

    for e in range(atr_window + win_min_bars, n, e_step):
        best = None
        for L in range(win_min_bars, win_max_bars+1, len_step):
            s = e - L + 1
            if s < atr_window: continue
            idxs = np.arange(s, e+1)
            hi_idx = idxs[is_h[s:e+1]]
            lo_idx = idxs[is_l[s:e+1]]
            if len(hi_idx) < 2 or len(lo_idx) < 2:
                k = min(4, len(idxs))
                if k < 2: continue
                top_hi = idxs[np.argsort(highs[s:e+1])[-k:]]
                bot_lo = idxs[np.argsort(lows[s:e+1])[:k]]
                hi_idx = np.sort(top_hi[:max(2, len(top_hi)//2)])
                lo_idx = np.sort(bot_lo[:max(2, len(bot_lo)//2)])
            uh_s, uh_b, uh_r2 = _fit_line(hi_idx, highs[hi_idx])
            lh_s, lh_b, lh_r2 = _fit_line(lo_idx, lows[lo_idx])
            price_scale = close[s:e+1].mean()
            uh_n = _norm_slope(uh_s, price_scale)
            lh_n = _norm_slope(lh_s, price_scale)
            mid_n = _norm_slope((uh_s + lh_s)/2.0, price_scale)
            if abs(uh_n) > flat_tol_norm or abs(lh_n) > flat_tol_norm: continue
            if (uh_r2 + lh_r2)/2.0 < r2_min: continue
            if abs(mid_n) > drift_tol_norm: continue
            width = _line_y(uh_s, uh_b, idxs) - _line_y(lh_s, lh_b, idxs)
            if np.any(width <= 0): continue
            w_mean = float(width.mean())
            w_std  = float(width.std(ddof=0))
            if w_mean > width_max_atr * atr[e]: continue
            w_stab = (w_std / max(w_mean, 1e-9))
            if w_stab > width_stability_max: continue
            tol = touch_tol_atr * atr[e]
            up_vals = _line_y(uh_s, uh_b, hi_idx)
            lo_vals = _line_y(lh_s, lh_b, lo_idx)
            touch_up = int(np.sum(np.abs(highs[hi_idx] - up_vals) <= tol))
            touch_lo = int(np.sum(np.abs(lows[lo_idx]  - lo_vals) <= tol))
            if touch_up < min_touches_each or touch_lo < min_touches_each: continue
            up_e = float(_line_y(uh_s, uh_b, e))
            lo_e = float(_line_y(lh_s, lh_b, e))
            mid_e = (up_e + lo_e) * 0.5
            expect = "up" if close[e] >= mid_e else "down"
            broken, b_idx, entry, stop = _confirm_break(e, expect, uh_s, uh_b, lh_s, lh_b)
            if require_breakout and not broken: continue
            height = float(width[0])
            if np.isnan(entry):
                entry = up_e if expect=="up" else lo_e
                stop  = lo_e - 0.25*atr[e] if expect=="up" else up_e + 0.25*atr[e]
            target = entry + height if expect=="up" else entry - height
            fit_q   = max(0.0, min(1.0, (uh_r2 + lh_r2)/2.0))
            flat_q  = 1.0 - min(1.0, max(abs(uh_n), abs(lh_n)) / flat_tol_norm)
            stab_q  = 1.0 - min(1.0, w_stab / width_stability_max)
            touch_q = min(1.0, 0.5*min(1.0, touch_up/min_touches_each) + 0.5*min(1.0, touch_lo/min_touches_each))
            drift_q = 1.0 - min(1.0, abs(mid_n)/drift_tol_norm)
            quality = float(np.clip(0.30*fit_q + 0.25*flat_q + 0.20*stab_q + 0.15*touch_q + 0.10*drift_q, 0, 1))
            pat = {"type":"rectangle","dir":"bull" if expect=="up" else "bear","start_idx":int(s),"end_idx":int(e),"breakout_idx":int(b_idx) if b_idx is not None else None,
                   "upper_line":(float(uh_s),float(uh_b),float(uh_r2)),"lower_line":(float(lh_s),float(lh_b),float(lh_r2)),
                   "width_mean":w_mean,"width_std":w_std,"width_stability":w_stab,"touches_upper":int(touch_up),"touches_lower":int(touch_lo),
                   "quality_score":quality,"entry":float(entry),"stop":float(stop),"target":float(target)}
            cand = (quality, pat)
            if best is None or cand[0] > best[0]: best = cand
        if best is not None: patterns.append(best[1])
    return patterns

def _measure(fn, df, repeat:int=1, **kwargs):
    t0 = time.perf_counter()
    pats: List[Dict] = []
    for _ in range(max(1,int(repeat))):
        pats = fn(df, **kwargs)
    dt = time.perf_counter() - t0
    q = np.mean([p.get("quality_score", np.nan) for p in pats]) if pats else float("nan")
    return dt, len(pats), q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(Path("data/USDJPY_15m.csv")))
    ap.add_argument("--last-N", type=int, default=2000)
    ap.add_argument("--win-max", type=int, default=48)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--baseline", action="store_true", help="旧アルゴリズムとの比較を表示")
    ap.add_argument("--parallel", action="store_true", help="最適化版を並列実行")
    ap.add_argument("--jobs", type=int, default=-1, help="並列ジョブ数 (joblib n_jobs)")
    ap.add_argument("--e-step", type=int, default=3, help="終端バーのステップ幅")
    ap.add_argument("--len-step", type=int, default=2, help="窓長ステップ幅")
    ap.add_argument("--backend", type=str, default="threads", choices=["threads","loky"], help="joblib backend")
    ap.add_argument("--use-numba", action="store_true", help="Numba最適化を有効化(実験的)")
    args = ap.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Ensure numeric cols and reset index
    for c in ("open","high","low","close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["high","low","close"]).reset_index(drop=True)

    dt_opt, n_opt, q_opt = _measure(
        detect_rectangles, df,
        repeat=int(args.repeat), last_N=int(args.last_N), win_max_bars=int(args.win_max), enable_low_pivot_fallback=True,
        e_step=int(args.e_step), len_step=int(args.len_step), use_parallel=bool(args.parallel), n_jobs=int(args.jobs), use_numba=bool(args.use_numba)
    )
    mode_txt = "parallel" if args.parallel else "serial"
    nb_txt = ", numba" if args.use_numba and HAVE_NUMBA else ""
    if args.use_numba and not HAVE_NUMBA:
        print("[warn] numba が見つかりませんでした。通常実装で実行します。")
    print(f"Optimized Rectangle ({mode_txt}{nb_txt}): time={dt_opt:.3f}s patterns={n_opt} avg_quality={q_opt:.3f}")
    if args.baseline:
        dt_base, n_base, q_base = _measure(
            detect_rectangles_baseline, df,
            repeat=int(args.repeat), last_N=int(args.last_N), win_max_bars=int(args.win_max), e_step=int(args.e_step), len_step=int(args.len_step)
        )
        ratio = dt_opt / dt_base if dt_base > 0 else float('nan')
        print(f"Baseline Rectangle:  time={dt_base:.3f}s patterns={n_base} avg_quality={q_base:.3f}")
        print(f"Speedup: x{(dt_base/dt_opt):.2f} (opt faster) | time_ratio_opt/base={ratio:.3f}")


if __name__ == "__main__":
    main()
