import numpy as np
import pandas as pd
import os
from pathlib import Path


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    a = a[m]
    b = b[m]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _maybe_dump_leakage_report(df: pd.DataFrame, rows: list[tuple], out_path: str | None) -> None:
    """Optionally dump suspects to CSV for debugging CI/local runs."""
    if not out_path:
        return
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rep = pd.DataFrame(rows, columns=["feature", "corr_now", "corr_fwd"])
    rep["delta_abs"] = (rep["corr_fwd"].abs() - rep["corr_now"].abs())
    rep = rep.sort_values("delta_abs", ascending=False)
    rep.to_csv(p, index=False, encoding="utf-8")


def test_no_obvious_future_leakage_vs_y():
    """Detect very obvious look-ahead leakage.

    This is a *sanity* test, not a proof:
    - For each feature, compare corr(feature_t, y_t) vs corr(feature_t, y_{t+1}).
    - If y_{t+1} correlates much more strongly than y_t across many features,
      it suggests the feature may be using future information.

    We keep thresholds loose to avoid false positives.
    """

    from ai_train_break import load_and_preprocess, get_label_config, make_dataset

    raw = load_and_preprocess("data/USDJPY_15m.csv")
    # minimal args emulation
    class _A:
        horizon_bars = 20
        buffer_ratio = 0.0015

    label_cfg = get_label_config(_A)
    df = make_dataset(raw, _A.horizon_bars, _A.buffer_ratio, label_cfg)

    # ensure chronological
    assert df["timestamp"].is_monotonic_increasing

    y = df["y"].astype(float).to_numpy()
    y_fwd = np.roll(y, -1)
    y_fwd[-1] = np.nan

    feature_cols = [c for c in df.columns if c not in {"timestamp", "open", "high", "low", "close", "volume", "y"}]

    # Focus on numeric features only
    bad = []
    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        c_now = _corr(x, y)
        c_fwd = _corr(x, y_fwd)
        if not (np.isfinite(c_now) and np.isfinite(c_fwd)):
            continue

        # If corr with future y is substantially higher than with current y, flag.
        # Use absolute corr to catch both directions.
        if abs(c_fwd) - abs(c_now) > 0.20 and abs(c_fwd) > 0.30:
            bad.append((c, c_now, c_fwd))

    # Allow a tiny number of spurious flags, but not many.
    # If it fails, show top suspects sorted by (|corr_fwd|-|corr_now|).
    if len(bad) > 2:
        bad_sorted = sorted(bad, key=lambda t: (abs(t[2]) - abs(t[1])), reverse=True)
        top = bad_sorted[:10]
        out_csv = os.getenv("LEAKAGE_REPORT_PATH", "")
        _maybe_dump_leakage_report(df, bad_sorted, out_csv)
        msg = "; ".join([f"{c}: corr_now={cn:+.3f}, corr_fwd={cf:+.3f}" for (c, cn, cf) in top])
        extra = f" (report: {out_csv})" if out_csv else ""
        raise AssertionError(f"potential look-ahead leakage features (top){extra}: {msg}")

