from __future__ import annotations
import os
import time
import yaml
import numpy as np
import pandas as pd

from constants import LOG_DIR, SIGNALS_LOG
from app.drift.drift_monitor import DriftMonitor


def _load_gate_drift_cfg() -> dict:
    try:
        y = yaml.safe_load(open("configs/drift.yml", "r", encoding="utf-8")) or {}
        return y
    except Exception:
        return {"drift": {"psi_warn": 0.25, "psi_halt": 0.5, "actions": {"warn": {"quality_add": 0.02, "p_add": 0.01}, "halt": {"entry_stop": True}}}}


def _select_series(df: pd.DataFrame) -> pd.Series:
    # Prefer calibrated probability if present, else raw
    for col in ("p_cal", "p_raw", "proba", "p"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            return s
    # Fallback: empty series
    return pd.Series([], dtype=float)


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    cfg = _load_gate_drift_cfg()
    dm = DriftMonitor(cfg)

    if not os.path.exists(SIGNALS_LOG):
        status = {"level": "ok", "adjust": {}, "metrics": {}, "ts": time.time()}
        yaml.safe_dump(status, open(os.path.join(LOG_DIR, "drift_status.yml"), "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
        print("[drift] no signals log -> ok")
        return 0

    df = pd.read_parquet(SIGNALS_LOG)
    if df.empty:
        status = {"level": "ok", "adjust": {}, "metrics": {}, "ts": time.time()}
        yaml.safe_dump(status, open(os.path.join(LOG_DIR, "drift_status.yml"), "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
        print("[drift] empty signals -> ok")
        return 0

    # Determine windows (by days if ts_jst present; else last N ratios)
    ts_col = None
    for c in ("ts_jst", "timestamp", "ts"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is not None:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col])
        except Exception:
            ts_col = None

    drift_cfg = (cfg or {}).get("drift", {})
    base_days = int((drift_cfg.get("windows") or {}).get("baseline_days", 30))
    roll_days = int((drift_cfg.get("windows") or {}).get("rolling_days", 7))

    if ts_col is not None:
        tmax = df[ts_col].max()
        recent_cut = tmax - pd.Timedelta(days=roll_days)
        base_cut = recent_cut - pd.Timedelta(days=base_days)
        cur_df = df[df[ts_col] > recent_cut]
        base_df = df[(df[ts_col] > base_cut) & (df[ts_col] <= recent_cut)]
    else:
        # Fallback: last N rows as current, previous chunk as baseline
        n = len(df)
        cur_n = min(2000, max(200, n // 10))
        cur_df = df.tail(cur_n)
        base_df = df.iloc[max(0, n - cur_n*6): max(0, n - cur_n)]

    p_base = _select_series(base_df).dropna().to_numpy()
    p_cur = _select_series(cur_df).dropna().to_numpy()

    res = dm.check_drift(p_base, p_cur, y_recent=None)
    status = {
        "level": res.level,
        "adjust": res.adjust,
        "metrics": {k: (float(v) if v is not None else None) for k, v in (res.metrics or {}).items()},
        "ts": time.time(),
        "counts": {"baseline": int(p_base.size), "current": int(p_cur.size)},
    }
    out_path = os.path.join(LOG_DIR, "drift_status.yml")
    yaml.safe_dump(status, open(out_path, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print("[drift] status ->", out_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
