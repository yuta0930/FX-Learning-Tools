# inference_break.py
import numpy as np
import pandas as pd
import joblib
import json

def load_break_model(model_path="models/break_model.joblib"):
    pkg = joblib.load(model_path)
    model = pkg["model"]
    use_cols = pkg.get("use_cols") or pkg.get("Xcols")  # 互換
    if use_cols is None:
        raise RuntimeError("models/break_model.joblib に use_cols/Xcols が見当たりません")
    return model, use_cols

def load_break_meta(meta_path="models/break_meta.json"):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _session_masks(df_feats: pd.DataFrame):
    # 1) ダミーがあれば最優先
    has_dummies = all(c in df_feats.columns for c in ["tokyo","london","ny"])
    if has_dummies:
        return {
            "Tokyo": (df_feats["tokyo"] > 0.5).values,
            "London": (df_feats["london"] > 0.5).values,
            "NY": (df_feats["ny"] > 0.5).values
        }
    # 2) 無ければ timestamp の hour で近似
    if "timestamp" not in df_feats.columns:
        raise RuntimeError("timestamp 列が無いためセッション判定ができません")
    h = pd.to_datetime(df_feats["timestamp"]).dt.hour
    return {
        "Tokyo": ((h>=9) & (h<15)).values,
        "London": ((h>=16) & (h<24)).values,
        "NY": ((h>=22) | (h<5)).values
    }

def predict_with_session_theta(df_feats: pd.DataFrame,
                               model,
                               use_cols,
                               meta: dict,
                               default_global=True):
    """
    return: DataFrame[timestamp, proba(0-1), theta, signal(0/1), session]
    """
    assert list(df_feats[use_cols].columns) == list(use_cols), \
        f"推論時の特徴量列が一致しません: {df_feats[use_cols].columns} vs {use_cols}"
    X = df_feats[use_cols].values.astype(float)
    proba = model.predict_proba(X)[:, 1].astype(float)

    # 念のためのクリップ（数値誤差対策）
    proba = np.clip(proba, 0.0, 1.0)

    th_global = float(meta.get("threshold", 0.5))
    sess_conf = meta.get("theta_by_session", {}) or {}
    masks = _session_masks(df_feats)

    theta_vec = np.full(len(df_feats), th_global, dtype=float) if default_global else np.full(len(df_feats), np.nan)
    session_vec = np.empty(len(df_feats), dtype=object); session_vec[:] = ""

    for name, m in masks.items():
        th = sess_conf.get(name, {}).get("theta", None)
        session_vec[m] = name
        if th is not None:
            theta_vec[m] = float(th)

    nan_idx = np.isnan(theta_vec)
    if nan_idx.any():
        theta_vec[nan_idx] = th_global
        session_vec[nan_idx] = "Global"

    signal = (proba >= theta_vec).astype(int)

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df_feats["timestamp"]).values,
        "proba": proba,        # 0〜1のまま
        "theta": theta_vec,
        "signal": signal,      # 0/1（UIで%表示に使わない）
        "session": session_vec
    })

    # デバッグ時の安全チェック
    assert out["proba"].between(0, 1).all(), "proba が 0〜1 の範囲外です（どこかで%化されている可能性）"
    return out
