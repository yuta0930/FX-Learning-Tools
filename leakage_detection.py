"""Temporal feature leakage detection utilities.

Purpose:
    Quickly flag features that may inadvertently contain future information
    (look-ahead bias) by comparing predictive power of current vs. shifted versions.

        Method:
        For each candidate feature column f:
            - Compute AUC/AP of (f, y)  -> current
            - Shift feature forward by 1 (f.shift(-1)) -> simulates leakage of one-step future ("future")
            - Shift feature backward by 1 (f.shift(1))  -> aligns a potentially future-looking feature with y ("past")
            - Recompute metrics (drop NaNs after shifting)
            - Suspicious if EITHER:
                    (A) future gain:   AUC_future - AUC_current >= auc_delta_min AND AUC_future >= auc_future_min
                    (B) past gain:     AUC_past   - AUC_current >= auc_delta_min AND AUC_past   >= auc_future_min
                (APでの条件でも可)。
                特に f ≈ y(t+1) 型の未来リーケージは "past" シフトで y(t) と強く整合しやすいため (B) が有効。

Returned DataFrame columns:
    feature, n_current, auc_current, ap_current,
    auc_future, ap_future, auc_past, ap_past,
    auc_future_gain, ap_future_gain, suspicious

Config (defaults inside function, may integrate with YAML later):
    min_samples: 300
    auc_delta_min: 0.05
    auc_future_min: 0.70
    ap_delta_min: 0.02
    ap_future_min: 0.10

Edge Cases:
    - Constant or near-constant features: skipped
    - Non-numeric columns: skipped
    - Insufficient samples after shift: skipped

Example:
    from leakage_detection import detect_temporal_leakage
    report = detect_temporal_leakage(df, y_col='y')
    print(report.query('suspicious'))
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import roc_auc_score, average_precision_score

_NUMERIC_KINDS = ("i", "u", "f")  # int, unsigned, float kinds


def _is_numeric(series: pd.Series) -> bool:
    return series.dtype.kind in _NUMERIC_KINDS


def _metric_safe(y_true, scores, func, default=np.nan):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return func(y_true, scores)
    except Exception:
        return default


def detect_temporal_leakage(
    df: pd.DataFrame,
    *,
    y_col: str = "y",
    timestamp_col: str = "timestamp",
    exclude: List[str] | None = None,
    min_samples: int = 300,
    auc_delta_min: float = 0.05,
    auc_future_min: float = 0.70,
    ap_delta_min: float = 0.02,
    ap_future_min: float = 0.10,
) -> pd.DataFrame:
    """Detect potential temporal leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Must include y_col and timestamp_col (monotonic increasing expected).
    y_col : str
        Target column name.
    exclude : list[str]
        Columns to forcibly skip besides y/timestamp.

    Returns
    -------
    pd.DataFrame
        Summary per feature with leakage suspicion flag.
    """
    if y_col not in df.columns:
        raise ValueError(f"y_col '{y_col}' not found")
    if timestamp_col not in df.columns:
        raise ValueError(f"timestamp_col '{timestamp_col}' not found")

    if not df[timestamp_col].is_monotonic_increasing:
        # sort defensively
        df = df.sort_values(timestamp_col).reset_index(drop=True)

    exclude = set(exclude or []) | {y_col, timestamp_col}

    y = df[y_col].values
    out_rows = []

    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        if not _is_numeric(s) or s.isna().all():
            continue
        arr = s.values.astype(float)
        if np.nanstd(arr) < 1e-10:
            continue  # constant

        # current
        mask_curr = np.isfinite(arr) & np.isfinite(y)
        if mask_curr.sum() < min_samples:
            continue
        y_curr = y[mask_curr]
        x_curr = arr[mask_curr]
        auc_curr = _metric_safe(y_curr, x_curr, roc_auc_score)
        ap_curr = _metric_safe(y_curr, x_curr, average_precision_score)

        # future (shift -1 brings future value earlier)
        arr_future = np.roll(arr, -1)
        arr_future[-1] = np.nan  # last becomes invalid
        mask_future = np.isfinite(arr_future) & np.isfinite(y)
        if mask_future.sum() < min_samples:
            continue
        y_future = y[mask_future]
        x_future = arr_future[mask_future]
        auc_future = _metric_safe(y_future, x_future, roc_auc_score)
        ap_future = _metric_safe(y_future, x_future, average_precision_score)

        # past (shift +1)
        arr_past = np.roll(arr, 1)
        arr_past[0] = np.nan
        mask_past = np.isfinite(arr_past) & np.isfinite(y)
        y_past = y[mask_past]
        x_past = arr_past[mask_past]
        auc_past = _metric_safe(y_past, x_past, roc_auc_score)
        ap_past = _metric_safe(y_past, x_past, average_precision_score)

        auc_gain = auc_future - auc_curr if np.isfinite(auc_future) and np.isfinite(auc_curr) else np.nan
        ap_gain = ap_future - ap_curr if np.isfinite(ap_future) and np.isfinite(ap_curr) else np.nan

        # 追加: "past" シフトでの顕著な改善もリーケージ疑いとみなす
        auc_past_gain = (auc_past - auc_curr) if np.isfinite(auc_past) and np.isfinite(auc_curr) else np.nan
        ap_past_gain  = (ap_past  - ap_curr)  if np.isfinite(ap_past)  and np.isfinite(ap_curr)  else np.nan

        suspicious_future = (
            (np.isfinite(auc_gain) and auc_gain >= auc_delta_min and (auc_future >= auc_future_min))
            or (np.isfinite(ap_gain)  and ap_gain  >= ap_delta_min  and (ap_future  >= ap_future_min))
        )
        suspicious_past = (
            (np.isfinite(auc_past_gain) and auc_past_gain >= auc_delta_min and (auc_past >= auc_future_min))
            or (np.isfinite(ap_past_gain)  and ap_past_gain  >= ap_delta_min  and (ap_past  >= ap_future_min))
        )

        suspicious = bool(suspicious_future or suspicious_past)

        out_rows.append({
            "feature": col,
            "n_current": int(mask_curr.sum()),
            "auc_current": float(auc_curr) if np.isfinite(auc_curr) else np.nan,
            "ap_current": float(ap_curr) if np.isfinite(ap_curr) else np.nan,
            "auc_future": float(auc_future) if np.isfinite(auc_future) else np.nan,
            "ap_future": float(ap_future) if np.isfinite(ap_future) else np.nan,
            "auc_past": float(auc_past) if np.isfinite(auc_past) else np.nan,
            "ap_past": float(ap_past) if np.isfinite(ap_past) else np.nan,
            "auc_future_gain": float(auc_gain) if np.isfinite(auc_gain) else np.nan,
            "ap_future_gain": float(ap_gain) if np.isfinite(ap_gain) else np.nan,
            "suspicious": bool(suspicious),
        })

    if not out_rows:
        return pd.DataFrame(columns=[
            "feature","n_current","auc_current","ap_current","auc_future","ap_future",
            "auc_past","ap_past","auc_future_gain","ap_future_gain","suspicious"
        ])

    rep = pd.DataFrame(out_rows).sort_values("auc_future_gain", ascending=False)
    return rep

if __name__ == "main":  # simple manual test
    # Basic smoke test using random data
    n = 1000
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(0)
    y = rng.integers(0,2,size=n)
    df = pd.DataFrame({
        "timestamp": ts,
        "y": y,
        "feat_noise": rng.normal(size=n),
        "feat_future_like": np.roll(y, -1) + rng.normal(scale=0.01,size=n),
    })
    r = detect_temporal_leakage(df)
    print(r.head())
