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
    add = pd.concat([
        df[["timestamp","ret_8","ret_12","atr14_norm","atr_ratio","d_atr14",
            "range_pct","body_ratio","wick_up_ratio","wick_dn_ratio",
            "slope_short_6","z_close_20"]].reset_index(drop=True),
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
    return tr.rolling(period).mean()
