from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml


def load_reason_map(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cats = data.get("categories") or {}
        return {str(k): [str(x) for x in (v or [])] for k, v in cats.items()}
    except Exception:
        return {}


def normalize_reasons_series(s: pd.Series, mapping: Dict[str, List[str]]) -> pd.Series:
    if s is None or not mapping:
        return s
    s = s.astype("string").fillna("")
    lower = s.str.lower()
    out = pd.Series([""] * len(s), index=s.index, dtype="string")
    for cat, patterns in mapping.items():
        for p in patterns:
            m = lower.str.contains(str(p).lower(), na=False)
            # 既に分類済みでないところのみ上書き
            out = out.mask((out == "") & m, cat)
    # 未分類は元の値を返す（空は空のまま）
    out = out.where(out != "", other=s)
    return out


def normalize_reasons_df(
    df: pd.DataFrame,
    *,
    reason_cols: Tuple[str, ...] = ("reason", "deny_reason", "gate_reason"),
    mapping: Dict[str, List[str]],
    out_col: str = "reason_norm",
) -> pd.DataFrame:
    if not mapping:
        return df
    col = next((c for c in reason_cols if c in df.columns), None)
    if col is None:
        return df
    out = df.copy()
    out[out_col] = normalize_reasons_series(out[col], mapping)
    return out
