import pandas as pd
import numpy as np

# --- level-relative features (shared for train/infer) ---
from typing import Optional, List, Dict
try:
    # utils.ta は本プロジェクト内ユーティリティ（lower-caseカラム前提）
    from utils.ta import swing_pivots, horizontal_levels, atr as _atr_ta
except Exception:  # 単体実行時のフォールバック（同等実装は下で定義済み）
    swing_pivots = None
    horizontal_levels = None

try:
    from featx import add_volatility_and_interactions as _add_vol_and_interactions
except Exception:
    _add_vol_and_interactions = None

def _safe_div(a, b, eps=1e-9):
    return np.where(np.abs(b) < eps, 0.0, a / b)

def _rolling_slope(x: pd.Series, win: int = 6) -> pd.Series:
    idx = np.arange(win, dtype=float)
    def _slope(arr):
        y = np.asarray(arr, dtype=float)
        x_ = idx
        x_mean = x_.mean(); y_mean = y.mean()
        num = ((x_ - x_mean) * (y - y_mean)).sum()
        den = ((x_ - x_mean)**2).sum()
        return 0.0 if den < 1e-9 else num / den
    return x.rolling(win).apply(_slope, raw=True)

def _zscore(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win).mean()
    v = s.rolling(win).std(ddof=0)
    return _safe_div(s - m, v)

def _hour_sin_cos(ts: pd.Series) -> pd.DataFrame:
    h = ts.dt.hour.astype(float)
    sin_h = np.sin(2*np.pi*h/24.0)
    cos_h = np.cos(2*np.pi*h/24.0)
    return pd.DataFrame({"sin_hour": sin_h, "cos_hour": cos_h})

def augment_features(feats: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    # normalize timezone to naive to avoid merge mismatch
    try:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if getattr(ts.dtype, 'tz', None) is not None:
            try:
                ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)
            except Exception:
                ts = ts.dt.tz_localize(None)
        df["timestamp"] = ts
    except Exception:
        pass
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # 基本派生
    df["ret_1"]  = df["close"].pct_change(1)
    df["ret_4"]  = df["close"].pct_change(4)
    df["ret_8"]  = df["close"].pct_change(8)
    df["ret_12"] = df["close"].pct_change(12)

    atr14 = _atr(df, 14)
    atr56 = _atr(df, 56)
    df["atr14_norm"] = _safe_div(atr14, df["close"].abs())
    df["atr_ratio"]   = _safe_div(atr14, atr56.replace(0, np.nan))
    df["d_atr14"]     = atr14.pct_change(1)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    wick_up = (df["high"] - df[["open","close"]].max(axis=1)).clip(lower=0)
    wick_dn = (df[["open","close"]].min(axis=1) - df["low"]).clip(lower=0)
    df["range_pct"]    = _safe_div(df["high"] - df["low"], df["close"].abs())
    df["body_ratio"]   = _safe_div(body, rng)
    df["wick_up_ratio"]= _safe_div(wick_up, rng)
    df["wick_dn_ratio"]= _safe_div(wick_dn, rng)

    slope6 = _rolling_slope(df["close"], 6)
    df["slope_short_6"] = _safe_div(slope6, df["close"].abs())

    df["z_close_20"] = _zscore(df["close"], 20)

    hc = _hour_sin_cos(df["timestamp"])
    # --- 上位足近似（1h/4h）---
    try:
        ts = pd.to_datetime(df["timestamp"])  # tz-aware/naiveどちらでもOK
        if len(ts) >= 3:
            deltas = ts.diff().dropna().dt.total_seconds().values
            step = float(np.median(deltas)) if len(deltas) else 900.0
            per_h = max(1, int(round(3600.0 / max(step, 1.0))))
        else:
            per_h = 4  # 15m相当のフォールバック
    except Exception:
        per_h = 4
    w1 = int(per_h)
    w4 = int(per_h * 4)
    def _roll_slope_norm(s: pd.Series, w: int) -> pd.Series:
        if w <= 1: return pd.Series([0.0]*len(s))
        idx = np.arange(w, dtype=float)
        def _sl(arr):
            y = np.asarray(arr, dtype=float)
            xm, ym = idx.mean(), y.mean()
            den = ((idx-xm)**2).sum()
            if den <= 1e-9: return 0.0
            return float(((idx-xm)*(y-ym)).sum()/den)
        sl = s.rolling(w).apply(_sl, raw=True)
        return _safe_div(sl, s.abs())
    # 1h, 4h 変化率とトレンド、ATRの相対量
    df["mtf1h_ret"]  = df["close"].pct_change(w1) if w1 < len(df) else 0.0
    df["mtf4h_ret"]  = df["close"].pct_change(w4) if w4 < len(df) else 0.0
    df["mtf1h_slope"] = _roll_slope_norm(df["close"], w1)
    df["mtf4h_slope"] = _roll_slope_norm(df["close"], w4)
    atr1h = _atr(df, max(2, w1))
    atr4h = _atr(df, max(4, w4))
    df["mtf1h_atr_norm"] = _safe_div(atr1h, df["close"].abs())
    df["mtf4h_atr_norm"] = _safe_div(atr4h, df["close"].abs())
    # one-hot風の方向フラグ
    df["mtf1h_up"] = (df["mtf1h_slope"] > 0).astype(float)
    df["mtf4h_up"] = (df["mtf4h_slope"] > 0).astype(float)
    add = pd.concat([
        df[["timestamp","ret_8","ret_12","atr14_norm","atr_ratio","d_atr14",
            "range_pct","body_ratio","wick_up_ratio","wick_dn_ratio",
            "slope_short_6","z_close_20",
            # MTF features
            "mtf1h_ret","mtf4h_ret","mtf1h_slope","mtf4h_slope",
            "mtf1h_atr_norm","mtf4h_atr_norm","mtf1h_up","mtf4h_up"
            ]].reset_index(drop=True),
        hc.reset_index(drop=True)
    ], axis=1)

    out = feats.merge(add, on="timestamp", how="left")

    if "touch_density" in out.columns and "atr14_norm" in out.columns:
        out["touch_x_atr"] = out["touch_density"] * out["atr14_norm"]
    if "slope_long" in out.columns and "ny" in out.columns:
        out["slope_long_x_ny"] = out["slope_long"] * out["ny"]

    # --- Align with training-side extended features (featx) ---
    # This adds volatility-normalized features (*_v), ATR regime dummies, interactions and light polynomials.
    if _add_vol_and_interactions is not None:
        try:
            out = _add_vol_and_interactions(out, raw_lc=df)
        except Exception:
            # best-effort; keep base features
            pass

    # --- Extra features referenced by trained models / smoke tests ---
    # Keep alignment by timestamp to avoid index drift.
    try:
        extra = pd.DataFrame({"timestamp": df["timestamp"]})
        extra["high_low_ratio_20"] = (
            (df["high"].rolling(20).max() / df["low"].rolling(20).min())
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
        )
        extra_atr = _atr(df, 14)
        extra["atr_change_10"] = extra_atr.pct_change(10).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = out.merge(extra, on="timestamp", how="left")
    except Exception:
        pass

    out = out.fillna(0.0)
    return out

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/float(period), adjust=False).mean()


# ============================================================
# Level-relative features
# ============================================================
def compute_level_relative_features(
    raw_lc: pd.DataFrame,
    *,
    window: int = 400,
    look_pivot: int = 11,
    min_samples: int = 4,
    eps: Optional[float] = None,
    k: int = 2,
    near_alpha_atr: float = 0.75,
    use_project_ta: bool = True,
    stride: int = 1,
) -> pd.DataFrame:
    """
    Compute level-relative features per bar using only past data.

    Inputs (lower-case columns expected): timestamp, open, high, low, close[, volume]
    Outputs (aligned by index):
      - lvl_k{i}_up, lvl_k{i}_dn: ATR-normalized distance to i-th nearest level above/below
      - lvl_near_cnt: #levels within near_thr (near_alpha_atr * ATR)
      - lvl_near_flag: 1 if lvl_near_cnt > 0
      - lvl_span_atr: (nearest_up - nearest_dn) / ATR (if both exist, else 0)

    Notes:
      - Uses project utils.ta.swing_pivots/horizontal_levels if available.
      - Avoids lookahead by building levels from a rolling past window [i-window+1, i].
      - k kept small for cost; window default ~400 bars.
    """
    df = raw_lc.copy().sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].is_monotonic_increasing
    n = len(df)
    if n == 0:
        cols = [f"lvl_k{i}_up" for i in range(1, k+1)] + [f"lvl_k{i}_dn" for i in range(1, k+1)] + [
            "lvl_near_cnt", "lvl_near_flag", "lvl_span_atr"
        ]
        return pd.DataFrame({c: [] for c in ["timestamp"] + cols})

    # ATR for normalization (use project ta if requested)
    try:
        atr14 = _atr_ta(df, 14) if (use_project_ta and _atr_ta is not None) else _atr(df, 14)
    except Exception:
        atr14 = _atr(df, 14)
    atr14 = atr14.ffill().fillna(atr14.median()).replace(0, np.nan)

    # Output arrays
    up = {i: np.zeros(n, dtype=float) for i in range(1, k+1)}
    dn = {i: np.zeros(n, dtype=float) for i in range(1, k+1)}
    near_cnt = np.zeros(n, dtype=float)
    span_atr = np.zeros(n, dtype=float)

    # Main loop (rolling window levels)
    stride = max(1, int(stride))
    last_levels: List[float] = []
    for i in range(n):
        recompute = (i % stride == 0) or (i == n - 1) or (not last_levels)
        if recompute:
            lo = max(0, i - int(window) + 1)
            sub = df.iloc[lo:i+1]
            try:
                if swing_pivots is not None and horizontal_levels is not None:
                    ph, pl = swing_pivots(sub, look_pivot)
                    q = {}
                    last_levels = horizontal_levels(ph, pl, eps=eps, min_samples=min_samples, quality_out=q)
                else:
                    std = float(np.std(sub["close"])) or 1e-6
                    step = max(std * 0.05, 1e-6)
                    vals = np.r_[sub["high"].values, sub["low"].values]
                    bins: Dict[int, List[float]] = {}
                    for v in vals:
                        b = int(round(v / step))
                        bins.setdefault(b, []).append(float(v))
                    last_levels = [float(np.mean(vs)) for vs in bins.values() if len(vs) >= min_samples]
                    last_levels = sorted(last_levels)
            except Exception:
                last_levels = []

        c = float(df["close"].iloc[i])
        a = float(atr14.iloc[i]) if not np.isnan(atr14.iloc[i]) else 1.0
        if a <= 1e-12:
            a = 1.0

        if not last_levels:
            for j in range(1, k+1):
                up[j][i] = 0.0
                dn[j][i] = 0.0
            near_cnt[i] = 0.0
            span_atr[i] = 0.0
            continue

        above = [lv for lv in last_levels if lv > c]
        below = [lv for lv in reversed(last_levels) if lv < c]

        for j in range(1, k+1):
            up[j][i] = max(0.0, (above[j-1] - c) / a) if j <= len(above) else 0.0
            dn[j][i] = max(0.0, (c - below[j-1]) / a) if j <= len(below) else 0.0

        near_thr = max(1e-8, near_alpha_atr * a)
        near_cnt[i] = float(sum(1 for lv in last_levels if abs(lv - c) <= near_thr))
        span_atr[i] = max(0.0, (above[0] - below[0]) / a) if (above and below) else 0.0

    # Assemble DataFrame
    cols = {
        **{f"lvl_k{i}_up": up[i] for i in up},
        **{f"lvl_k{i}_dn": dn[i] for i in dn},
        "lvl_near_cnt": near_cnt,
        "lvl_near_flag": (near_cnt > 0).astype(float),
        "lvl_span_atr": span_atr,
    }
    out = pd.DataFrame({"timestamp": df["timestamp"].values, **cols})
    # Fill any residual NaN with 0.0 for model-friendliness
    return out.fillna(0.0)
