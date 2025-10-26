from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

from .observability import load_logs
from src.policy.reasons_normalize import load_reason_map, normalize_reasons_df
from src.core.drift import window_drift


@dataclass
class HealthThresholds:
    lookback_days: int = 1
    # 直近期間のブロック率がこれを超えたらNG
    max_block_rate: float = 0.8
    # 正規化できない理由の比率がこれを超えたらNG（マップ存在時）
    max_unknown_ratio: float = 0.3
    # ドリフトのPSI/JSがこれを超えたらNG（いずれか設定時にドリフト評価を有効化）
    max_drift_psi: Optional[float] = None
    max_drift_js: Optional[float] = None


def _block_rate(df: pd.DataFrame, since: datetime) -> float:
    if df.empty:
        return 0.0
    if "time" in df.columns:
        df = df[pd.to_datetime(df["time"], errors="coerce") >= since]
    if df.empty or "trade_ok" not in df.columns:
        return 0.0
    total = len(df)
    blocked = (df["trade_ok"].astype(bool) == False).sum()  # noqa: E712
    return (blocked / total) if total else 0.0


def _unknown_ratio(df: pd.DataFrame, mapping_path: Optional[Path]) -> Optional[float]:
    if not mapping_path:
        return None
    mapping = load_reason_map(mapping_path)
    if not mapping:
        return None
    df2 = normalize_reasons_df(df, mapping=mapping)
    col_src = next((c for c in ["reason", "deny_reason", "gate_reason"] if c in df2.columns), None)
    if col_src is None or "reason_norm" not in df2.columns:
        return None
    mask_src = df2[col_src].astype(str).ne("")
    total = mask_src.sum()
    if total == 0:
        return 0.0
    unknown = (df2.loc[mask_src, "reason_norm"] == df2.loc[mask_src, col_src]).sum()
    return unknown / total


def assess_health(
    paths: List[Path],
    th: HealthThresholds,
    reasons_map: Optional[Path] = None,
    *,
    drift_source: Optional[Path] = None,
    drift_column: str = "close",
    drift_ref_n: int = 2000,
    drift_cur_n: int = 500,
    drift_bins: int = 20,
) -> Dict[str, Any]:
    df = load_logs(paths)
    since = datetime.now() - timedelta(days=th.lookback_days)
    br = _block_rate(df, since)
    ur = _unknown_ratio(df, reasons_map)
    # 品質ゲート由来のブロック率（列があればオプションで算出）
    br_quality = None
    blocks_quality_n = None
    total_n = None
    try:
        dff = df.copy()
        if "time" in dff.columns:
            dff = dff[pd.to_datetime(dff["time"], errors="coerce") >= since]
        if not dff.empty and ("trade_ok" in dff.columns):
            total_n = int(len(dff))
            blocks_total_n = int((dff["trade_ok"].astype(bool) == False).sum())  # noqa: E712
            if any(c in dff.columns for c in ("reason", "deny_reason", "gate_reason")):
                # unify_reasons_df は load_logs 内で適用済みのため 'reason' を優先
                col = "reason" if "reason" in dff.columns else ("deny_reason" if "deny_reason" in dff.columns else "gate_reason")
                mask_q = dff[col].astype(str).str.contains("pattern_quality<thr", case=False, na=False)
                blocks_quality_n = int(mask_q.sum())
                br_quality = float(blocks_quality_n / total_n) if total_n else 0.0
            else:
                br_quality = None
        else:
            br_quality = None
    except Exception:
        br_quality = None
        blocks_quality_n = None
        total_n = None
    # ドリフト（任意）
    drift: Optional[Dict[str, Any]] = None
    drift_ok = True
    if drift_source and (th.max_drift_psi is not None or th.max_drift_js is not None):
        try:
            # 価格などの系列を読み込み
            suf = drift_source.suffix.lower()
            if suf == ".parquet":
                sdf = pd.read_parquet(drift_source)
            else:
                sdf = pd.read_csv(drift_source)
            if drift_column in sdf.columns:
                metrics = window_drift(pd.to_numeric(sdf[drift_column], errors="coerce"), ref_n=int(drift_ref_n), cur_n=int(drift_cur_n), bins=int(drift_bins))
                drift = metrics
                # しきい値判定（設定されたもののみ）
                psi_ok = True if th.max_drift_psi is None else (float(metrics.get("psi", float("nan"))) <= th.max_drift_psi)
                js_ok = True if th.max_drift_js is None else (float(metrics.get("js", float("nan"))) <= th.max_drift_js)
                drift_ok = bool(psi_ok and js_ok)
            else:
                drift = {"error": f"column '{drift_column}' not found"}
        except Exception as e:
            drift = {"error": str(e)}
            # ドリフト計算失敗はゲートに影響させない（安全にスキップ）
            drift_ok = True
    ok = (br <= th.max_block_rate) and (ur is None or ur <= th.max_unknown_ratio) and drift_ok
    return {
        "ok": bool(ok),
        "lookback_days": th.lookback_days,
        "block_rate": br,
        **({"block_rate_quality": br_quality} if br_quality is not None else {}),
        **({"blocks_quality_n": blocks_quality_n} if blocks_quality_n is not None else {}),
        "unknown_ratio": ur,
        "drift": drift,
        "thresholds": th.__dict__,
        "inputs": [str(p) for p in paths],
        "ts": datetime.now().isoformat(),
    }
