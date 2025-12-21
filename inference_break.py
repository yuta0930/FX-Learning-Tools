# inference_break.py
import numpy as np
import pandas as pd
import joblib
import sys
from model_wrappers import TemperatureScaledModel as _TempModel  # ensure module is importable for pickle
import json
import os
import hashlib


def _estimate_bars_per_day(ts: pd.Series) -> int:
    """Estimate bars/day from timestamp median step.

    Mirrors training-side estimation to detect train/serve skew.
    """
    s = pd.to_datetime(ts, errors="coerce").dropna()
    if len(s) < 3:
        return 96
    deltas = s.diff().dropna().dt.total_seconds().values
    if len(deltas) == 0:
        return 96
    step = float(np.median(deltas))
    if step <= 0:
        return 96
    bpd = int(round(86400.0 / step))
    return max(1, min(24 * 60, bpd))


def _check_bars_per_day_consistency(df_feats: pd.DataFrame, meta: dict) -> str | None:
    """Warn (or raise) if bars/day differs between training meta and inference input.

    Returns:
        warning message (str) if mismatch is detected in non-strict mode, else None.
    """
    if not isinstance(meta, dict):
        return None
    expected = meta.get("bars_per_day_est")
    if expected is None:
        return None
    if "timestamp" not in df_feats.columns:
        return None

    expected_i = int(expected)
    current_i = int(_estimate_bars_per_day(df_feats["timestamp"]))
    if expected_i <= 0 or current_i <= 0:
        return None

    # tolerate small differences (DST / slight irregularities), but flag clear mismatches
    ratio = current_i / float(expected_i)
    ok = 0.8 <= ratio <= 1.25
    if ok:
        return None

    msg = (
        f"bars/day mismatch: meta bars_per_day_est={expected_i}, "
        f"inference bars_per_day_est={current_i} (ratio={ratio:.2f}). "
        "This may indicate timeframe mismatch or irregular timestamps."
    )
    strict = str(os.getenv("STRICT_BARS_PER_DAY", "0")).lower() in {"1", "true", "yes", "y"}
    if strict:
        raise RuntimeError(msg)
    print(f"[warn] {msg}")
    return msg

def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def load_break_model(model_path="models/break_model.joblib"):
    """Load trained model with backward-compat for legacy pickles.

    Ensures that references to __main__._TemperatureScaledModel can be resolved
    by exposing model_wrappers.TemperatureScaledModel under that name.
    """
    # Back-compat shim: expose symbol for legacy pickles
    # This lets pickle find __main__._TemperatureScaledModel at load time
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "_TemperatureScaledModel"):
        setattr(main_mod, "_TemperatureScaledModel", _TempModel)
    # Prefer calibrated model if exists and not explicitly overridden
    if model_path == "models/break_model.joblib":
        cal_path = "models/break_model_calibrated.joblib"
        if os.path.exists(cal_path):
            model_path = cal_path
    # Optional signature verification (soft by default)
    meta_path = "models/break_meta.json"
    meta_sig = None
    try:
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_sig = json.load(f)
    except Exception:
        meta_sig = None
    # Load model
    pkg = joblib.load(model_path)
    model = pkg["model"]
    use_cols = pkg.get("use_cols") or pkg.get("Xcols")  # 互換
    if use_cols is None:
        raise RuntimeError("models/break_model.joblib に use_cols/Xcols が見当たりません")
    # Verify signature if available
    try:
        strict = str(os.getenv("STRICT_MODEL_SIGNATURE", "0")).lower() in {"1","true","yes","y"}
        current_hash = _sha256_of_file(model_path)
        expected = None
        if isinstance(meta_sig, dict):
            expected = meta_sig.get("model_sha256")
        if expected and expected != current_hash:
            msg = f"Model signature mismatch: expected {expected[:8]}..., got {current_hash[:8]}..."
            if strict:
                raise RuntimeError(msg)
            else:
                print(f"[warn] {msg}")
    except Exception as _e:
        # Only warn on verification issues unless strict
        if str(os.getenv("STRICT_MODEL_SIGNATURE", "0")).lower() in {"1","true","yes","y"}:
            raise
        else:
            print(f"[warn] signature verification skipped: {_e}")
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
    _check_bars_per_day_consistency(df_feats, meta)

    # --- Feature alignment (train/serve skew guard) ---
    # Some sklearn estimators/pipelines expose feature_names_in_. Prefer that over caller-provided use_cols.
    # This prevents "X has N features, but StandardScaler is expecting M" when one extra column leaks in.
    expected_cols = None
    try:
        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            expected_cols = list(model.feature_names_in_)
    except Exception:
        expected_cols = None
    if expected_cols is None:
        # Try common pipeline step (e.g., scaler)
        for attr in ("named_steps",):
            try:
                steps = getattr(model, attr, None)
                if isinstance(steps, dict):
                    scaler = steps.get("scaler")
                    if scaler is not None and hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
                        expected_cols = list(scaler.feature_names_in_)
                        break
            except Exception:
                pass

    eff_cols = list(expected_cols) if expected_cols else list(use_cols)

    # If we still don't know exact names, at least match expected feature count.
    # CalibratedClassifierCV / wrappers may hide the underlying scaler's feature_names_in_.
    def _get_expected_n_features(m):
        for attr in ("n_features_in_",):
            try:
                v = getattr(m, attr, None)
                if isinstance(v, (int, np.integer)) and int(v) > 0:
                    return int(v)
            except Exception:
                pass
        # Try nested estimators commonly used by sklearn
        for nested_attr in ("base_estimator", "estimator", "calibrated_classifiers_", "classifier", "model"):
            try:
                nested = getattr(m, nested_attr, None)
            except Exception:
                nested = None
            if nested is None:
                continue
            # calibrated_classifiers_ is a list of internal calibrated estimators
            if isinstance(nested, list) and nested:
                for item in nested:
                    n = _get_expected_n_features(item)
                    if n is not None:
                        return n
            else:
                n = _get_expected_n_features(nested)
                if n is not None:
                    return n
        return None

    expected_n = _get_expected_n_features(model)
    if expected_cols is None and expected_n is not None and len(eff_cols) != expected_n:
        # Heuristic: drop trailing extras (most common when a new feature was appended).
        if len(eff_cols) > expected_n:
            dropped = eff_cols[expected_n:]
            eff_cols = eff_cols[:expected_n]
            print(f"[warn] adjusted use_cols by expected n_features_in_={expected_n}; dropped={dropped}")

    # Make a defensive copy to avoid mutating caller frame.
    dfX = df_feats.copy()
    missing = [c for c in eff_cols if c not in dfX.columns]
    extra = [c for c in dfX.columns if c not in eff_cols and c != "timestamp"]
    if missing:
        # Fill missing features with 0.0 (safe default). If this happens frequently, retraining is recommended.
        for c in missing:
            dfX[c] = 0.0
    # Drop extras (except timestamp used for output/session inference)
    if extra:
        dfX = dfX.drop(columns=extra)

    if missing or extra:
        # Soft visibility: helps diagnose why probabilities become flat/NaN after alignment.
        # Keep as print to avoid hard dependency on streamlit logger.
        print(
            f"[warn] feature alignment applied: missing_filled={len(missing)} extra_dropped={len(extra)} "
            f"(missing={missing[:5]}{'...' if len(missing)>5 else ''}, "
            f"extra={extra[:5]}{'...' if len(extra)>5 else ''})"
        )

    # Reorder and validate
    if list(dfX[eff_cols].columns) != list(eff_cols):
        raise AssertionError(
            "推論時の特徴量列が一致しません: "
            f"expected={eff_cols} actual={list(dfX[eff_cols].columns)} "
            f"(missing_filled={missing}, extra_dropped={extra})"
        )
    X = dfX[eff_cols].values.astype(float)
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
