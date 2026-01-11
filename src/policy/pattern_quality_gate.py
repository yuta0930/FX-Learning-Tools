from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd


def _safe_load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data or {}
    except Exception:
        return {}


def load_thresholds_yaml(path: Path = Path("config/patterns_quality_thresholds.yml")) -> Dict[str, Any]:
    """
        Example YAML (flat):

            Tokyo: 0.62
            London: 0.58
            NewYork: 0.60
            default: 0.58

        Or nested (pattern-scoped):

            triangle:
                Tokyo: 0.62
                London: 0.58
            rectangle:
                London: 0.12
            default:
                default: 0.55

        If missing or load fails, returns {} (No-Op). Values are coerced to float where possible.
    """
    raw = _safe_load_yaml(path)
    def _coerce_map(obj: Any) -> Any:
        # float or nested dicts of floats
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for kk, vv in obj.items():
                if isinstance(vv, dict):
                    out[str(kk)] = _coerce_map(vv)
                else:
                    try:
                        if vv is None:
                            continue
                        out[str(kk)] = float(vv)  # type: ignore
                    except Exception:
                        # ignore non-coercible
                        pass
            return out
        return obj

    if isinstance(raw, dict):
        return _coerce_map(raw)
    return {}


def enforce_pattern_quality_gate_df(
    df: pd.DataFrame,
    *,
    thresholds: Optional[Dict[str, Any]] = None,
    quality_col: str = "pattern_quality",
    session_col: str = "session",
    pattern_col: Optional[str] = None,
    reason_col: str = "deny_reason",
    logs_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply session/default-based quality thresholds to a DataFrame.

    - If thresholds is empty or quality column not present -> No-Op
    - Supports flat thresholds (session->thr) or nested (pattern->session->thr)
    - Uses session-specific threshold if available, else 'default' if present
    - Marks trade_ok False when quality < threshold and appends reason
    """
    if df is None or len(df) == 0:
        return df
    # Accept common aliases if primary column is missing (No-Op if none exist)
    qcol = quality_col
    if qcol not in df.columns:
        for alt in ("quality", "pattern_q", "q"):
            if alt in df.columns:
                qcol = alt
                break
        else:
            return df

    thr = thresholds or {}
    if not thr:
        return df

    out = df.copy()
    if "trade_ok" not in out.columns:
        out["trade_ok"] = True
    if reason_col not in out.columns:
        out[reason_col] = ""

    # Detect pattern column if nested thresholds provided
    pcol = pattern_col
    if pcol is None:
        for cand in ("pattern", "kind", "pattern_name", "pattern_type"):
            if cand in out.columns:
                pcol = cand
                break

    def _thr_for_session(sess: Optional[str]) -> Optional[float]:
        # If nested dicts: look up by pattern first
        try:
            if isinstance(thr, dict) and any(isinstance(v, dict) for v in thr.values()):
                pat_name = str(out[pcol].iloc[0]) if (pcol and pcol in out.columns and len(out)) else None
                # Try specific pattern mapping
                if pat_name and isinstance(thr.get(pat_name), dict):
                    tmap = thr.get(pat_name) or {}
                    if sess is not None and sess in tmap:
                        return float(tmap[sess])
                    return (tmap.get("default") if isinstance(tmap, dict) else None)
                # Try 'default' pattern mapping
                if isinstance(thr.get("default"), dict):
                    dmap = thr.get("default") or {}
                    if sess is not None and sess in dmap:
                        return float(dmap[sess])
                    return (dmap.get("default") if isinstance(dmap, dict) else None)
        except Exception:
            pass
        # Flat mapping fallback
        if sess is not None and sess in thr:
            try:
                return float(thr[sess])  # type: ignore
            except Exception:
                return None
        try:
            return float(thr.get("default", None))  # type: ignore
        except Exception:
            return None

    sess_vals = out[session_col].astype(str).to_numpy() if session_col in out.columns else np.array([None] * len(out))
    qvals = pd.to_numeric(out[qcol], errors="coerce").astype(float).to_numpy()
    thrs = np.array([_thr_for_session(s) for s in sess_vals], dtype="float64")

    mask_valid = ~np.isnan(thrs)
    mask_block = (qvals < thrs) & mask_valid

    if mask_block.any():
        out.loc[mask_block, "trade_ok"] = False
        # Append reason; create if missing
        def _append_reason(prev: str) -> str:
            prev = "" if (prev is None or (isinstance(prev, float) and np.isnan(prev))) else str(prev)
            base = " | ".join([p for p in [prev, "pattern_quality<thr"] if p and p.strip()])
            return base if base else "pattern_quality<thr"

        out.loc[mask_block, reason_col] = out.loc[mask_block, reason_col].astype("object").map(_append_reason)

        # Optional: record a compact failure event for observability
        try:
            if logs_dir:
                from src.monitoring.failure_log import log_failure  # local import to avoid hard dep at import time

                details = {
                    "blocked_count": int(mask_block.sum()),
                    "total": int(len(out)),
                    "sessions": (
                        out.loc[mask_block, session_col].astype(str).value_counts().to_dict()
                        if session_col in out.columns
                        else {}
                    ),
                }
                if pcol and pcol in out.columns:
                    details["patterns"] = out.loc[mask_block, pcol].astype(str).value_counts().to_dict()
                log_failure(
                    logs_dir=logs_dir,
                    reason="pattern_quality<thr",
                    details=details,
                    hint="Review weekly thresholds or lower session default; run scripts/weekly_quality_gate.py",
                )
        except Exception:
            # Never fail trading pipeline due to logging
            pass

    # Ensure scalar values are plain Python bool (tests use `is False`)
    try:
        out["trade_ok"] = out["trade_ok"].map(lambda v: bool(v)).astype(object)
    except Exception:
        pass

    return out
