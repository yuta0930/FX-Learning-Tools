import pandas as pd
import numpy as np

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
    assert df["timestamp"].is_monotonic_increasing

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

    out = out.fillna(0.0)
    return out

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/float(period), adjust=False).mean()
