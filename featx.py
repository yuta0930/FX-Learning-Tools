# tools/featx.py
import numpy as np
import pandas as pd

EPS = 1e-9

def _safe_div(a, b):
    return np.where(np.abs(b) < EPS, 0.0, a / b)

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/float(period), adjust=False).mean()

def _rv_sigma(df: pd.DataFrame, win: int = 20) -> pd.Series:
    ret1 = df["close"].pct_change(1)
    return ret1.rolling(win, min_periods=win).std(ddof=0)

def _qcut_one(s: pd.Series, q=(0.33, 0.66), labels=("low","mid","high")):
    try:
        return pd.qcut(s, q=[0.0, q[0], q[1], 1.0], labels=labels, duplicates="drop")
    except Exception:
        return pd.Series(index=s.index, dtype="category")

def add_volatility_and_interactions(
    feats: pd.DataFrame,
    raw_lc: pd.DataFrame,  # lower-case: timestamp, open, high, low, close, volume
    *,
    enable_poly: bool = True
) -> pd.DataFrame:
    df = raw_lc.copy().sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].is_monotonic_increasing

    # === ボラ正規化ブロック ===
    atr14 = _atr(df, 14)
    atr56 = _atr(df, 56)
    rv20  = _rv_sigma(df, 20)

    # 価格スケールを除いた「単位ボラ」
    vol_unit = _safe_div(atr14, df["close"].abs())
    vol_unit2 = rv20.replace(0, np.nan)

    # featsのtimestamp列でaddを揃える（indexではなく列で）
    # NOTE: Do NOT align by row index. `feats` may be filtered (dropna) and indices can drift.
    add = pd.DataFrame({"timestamp": feats["timestamp"]})

    # Prepare a timestamp-indexed lookup for raw-derived series
    df_ts = df[["timestamp"]].copy()
    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"], errors="coerce")
    _vol = pd.DataFrame(
        {
            "timestamp": df_ts["timestamp"],
            "vol_unit": pd.Series(vol_unit).astype(float).values,
            "atr14": atr14.astype(float).values,
            "atr56": atr56.astype(float).values,
            "rv20": rv20.astype(float).values,
        }
    ).dropna(subset=["timestamp"])
    _vol = _vol.drop_duplicates(subset=["timestamp"], keep="last")
    _vol = _vol.set_index("timestamp").sort_index()

    feats_ts = pd.to_datetime(add["timestamp"], errors="coerce")
    vol_unit_aligned = _vol.reindex(feats_ts)["vol_unit"].values
    atr14_aligned = _vol.reindex(feats_ts)["atr14"].values
    atr56_aligned = _vol.reindex(feats_ts)["atr56"].values
    rv20_aligned = _vol.reindex(feats_ts)["rv20"].values
    # 既存の一部特徴をボラで割る（相場レジームの影響を薄める）
    # feats側にあれば正規化版を作る
    def _maybe_norm(col, name):
        if col in feats.columns:
            add[name] = _safe_div(feats[col].values, (vol_unit_aligned + 1e-8))
    _maybe_norm("ret_1",  "ret_1_v")
    _maybe_norm("ret_4",  "ret_4_v")
    _maybe_norm("ret_8",  "ret_8_v")
    _maybe_norm("ret_12", "ret_12_v")
    _maybe_norm("range_pct", "range_pct_v")
    _maybe_norm("slope_short_6", "slope_short_6_v")

    # ATR/長期ATR 比、RV の追加
    # NOTE: Keep alignment on timestamp. For close we use feats-side close if present.
    if "close" in feats.columns:
        close_aligned = feats["close"].abs().values
    else:
        # fallback: raw close aligned on timestamp (may be NaN if mismatch)
        close_aligned = _vol.reindex(feats_ts)["atr14"].values * 0.0 + np.nan
    add["atr14_norm_v"] = _safe_div(atr14_aligned, np.abs(close_aligned))
    add["atr_ratio_56"] = _safe_div(atr14_aligned, np.where(np.abs(atr56_aligned) < 1e-12, np.nan, atr56_aligned))
    add["rv20"] = np.nan_to_num(rv20_aligned, nan=0.0)

    # ATRレジーム（3分位）
    atr_reg = _qcut_one(pd.Series(atr14_aligned).fillna(np.nanmedian(atr14_aligned)))
    for lab in ("low","mid","high"):
        add[f"reg_atr_{lab}"] = (atr_reg.astype(str) == lab).astype(float)

    # === 交互作用（軽量・解釈しやすいものだけ） ===
    out = feats.merge(add, on="timestamp", how="left")

    def _mul_if(a, b, name):
        if a in out.columns and b in out.columns:
            out[name] = out[a] * out[b]

    # 時間帯 × 歪み/近傍
    for sess in ("tokyo","london","ny"):
        if sess in out.columns:
            _mul_if(sess, "range_pct",   f"{sess}_x_range")
            _mul_if(sess, "touch_density", f"{sess}_x_touch")

    # 近傍×勢い/歪み
    _mul_if("touch_density", "atr14_norm_v", "touch_x_atr_v")
    _mul_if("touch_density", "slope_short_6_v", "touch_x_slope_v")
    _mul_if("z_close_20",    "atr14_norm_v", "z_x_atr_v")

    # レジーム×既存確率の効きやすい特徴
    for lab in ("low","mid","high"):
        _mul_if(f"reg_atr_{lab}", "z_close_20", f"reg_{lab}_x_z")
        _mul_if(f"reg_atr_{lab}", "slope_short_6_v", f"reg_{lab}_x_slope_v")

    # 軽い二次（必要時のみ）
    if enable_poly:
        for c in ("z_close_20","slope_short_6_v","ret_8_v","ret_12_v"):
            if c in out.columns:
                out[c + "_sq"] = (out[c] ** 2)

    return out.fillna(0.0)
