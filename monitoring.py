# monitoring.py
from __future__ import annotations
import os, json, time, math, traceback, threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# ====== 1) PSI（Population Stability Index） ======
def _hist_bins_from_probs(p: np.ndarray, n_bins: int = 10) -> np.ndarray:
    # 等頻度ベースの境界（学習OOFから作るのが望ましい）
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.unique(np.quantile(p, qs))
    if len(edges) < 3:
        edges = np.linspace(0, 1, n_bins+1)
    return edges

def compute_psi(baseline: np.ndarray, current: np.ndarray, *, edges: Optional[np.ndarray] = None,
                eps: float = 1e-6) -> Tuple[float, np.ndarray]:
    if edges is None:
        edges = _hist_bins_from_probs(baseline, n_bins=10)
    b = np.histogram(baseline, bins=edges)[0].astype(float)
    c = np.histogram(current,  bins=edges)[0].astype(float)
    b = b / max(b.sum(), eps)
    c = c / max(c.sum(), eps)
    psi_terms = (c - b) * np.log((c + eps) / (b + eps))
    return float(psi_terms.sum()), edges

def psi_severity(psi: float) -> str:
    # <0.1: 安定, 0.1-0.25: 注意, >0.25: ドリフト
    if psi < 0.10: return "OK"
    if psi < 0.25: return "WARN"
    return "DRIFT"

# ====== 2) 直近セッションの「閾値超過率」 ======
def threshold_exceed_rate(probs: np.ndarray, theta: float) -> float:
    if probs.size == 0: return float("nan")
    return float((probs >= theta).mean())

# ====== 3) ヘルスチェック ======
@dataclass
class HealthReport:
    ok: bool
    details: Dict[str, str]
    ts_utc: float

def healthcheck(*, model_path: str, meta_path: str, windows_df: Optional[pd.DataFrame],
                last_price_ts: Optional[pd.Timestamp], max_age_min: int = 5) -> HealthReport:
    det: Dict[str, str] = {}
    ok = True
    # 1) モデル/メタ
    try:
        ok_model = os.path.exists(model_path) and (os.path.getsize(model_path) > 0)
        det["model"] = "OK" if ok_model else "MISSING"
        ok &= ok_model
    except Exception as e:
        det["model"] = f"ERR:{e}"; ok = False

    try:
        ok_meta = os.path.exists(meta_path) and json.load(open(meta_path, "r", encoding="utf-8")) is not None
        det["meta"] = "OK" if ok_meta else "MISSING"
        ok &= ok_meta
    except Exception as e:
        det["meta"] = f"ERR:{e}"; ok = False

    # 2) イベント窓
    try:
        if windows_df is None or windows_df.empty:
            det["event_windows"] = "EMPTY"
        else:
            ok_cols = all(c in windows_df.columns for c in ("start","end"))
            det["event_windows"] = "OK" if ok_cols else "MISSING_COLS"
            ok &= ok_cols
    except Exception as e:
        det["event_windows"] = f"ERR:{e}"; ok = False

    # 3) 価格の鮮度
    try:
        if last_price_ts is None:
            det["price_feed"] = "NO_TS"
            ok = False
        else:
            age_min = (pd.Timestamp.utcnow().tz_localize("UTC") - last_price_ts.tz_convert("UTC")).total_seconds() / 60.0
            det["price_feed"] = f"{age_min:.1f}min"
            if age_min > max_age_min:
                ok = False
                det["price_feed"] += " (STALE)"
    except Exception as e:
        det["price_feed"] = f"ERR:{e}"; ok = False

    return HealthReport(ok=ok, details=det, ts_utc=time.time())

# ====== 4) 例外に強いセーフラッパ ======
def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, f"{e.__class__.__name__}: {e}"

# ====== 5) 簡易リトライ（指数バックオフ） ======
def retry(fn, *args, tries: int = 3, base_delay: float = 0.4, **kwargs):
    err = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = e
            time.sleep(base_delay * (2 ** i))
    raise err
