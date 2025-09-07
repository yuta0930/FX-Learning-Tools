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
    return tr.rolling(period, min_periods=period).mean()

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
    add = pd.DataFrame({"timestamp": feats["timestamp"]})
    # 既存の一部特徴をボラで割る（相場レジームの影響を薄める）
    # feats側にあれば正規化版を作る
    def _maybe_norm(col, name):
        if col in feats.columns:
            # vol_unitをfeatsのインデックスに合わせて揃える
            vol_unit_aligned = pd.Series(vol_unit, index=df.index).reindex(feats.index).values
            add[name] = _safe_div(feats[col].values, (vol_unit_aligned + 1e-8))
    _maybe_norm("ret_1",  "ret_1_v")
    _maybe_norm("ret_4",  "ret_4_v")
    _maybe_norm("ret_8",  "ret_8_v")
    _maybe_norm("ret_12", "ret_12_v")
    _maybe_norm("range_pct", "range_pct_v")
    _maybe_norm("slope_short_6", "slope_short_6_v")

    # ATR/長期ATR 比、RV の追加
    add["atr14_norm_v"] = _safe_div(
        atr14.reindex(add.index),
        df["close"].abs().reindex(add.index)
    )
    add["atr_ratio_56"] = _safe_div(
        atr14.reindex(add.index),
        atr56.replace(0, np.nan).reindex(add.index)
    )
    add["rv20"]         = rv20.fillna(0.0).reindex(add.index)

    # ATRレジーム（3分位）
    atr_reg = _qcut_one(atr14.fillna(atr14.median())).reindex(add.index)
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
