from __future__ import annotations

import pandas as pd
from pathlib import Path
import yaml


_DEFAULT = {"Tokyo": (9, 15), "London": (16, 20), "NewYork": (21, 5)}


def _load_cfg(path: Path = Path("config/session.yml")) -> dict[str, tuple[int, int]]:
    try:
        if not path.exists():
            return _DEFAULT
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out: dict[str, tuple[int, int]] = {}
        for k, v in data.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                out[str(k)] = (int(v[0]), int(v[1]))
        return out or _DEFAULT
    except Exception:
        return _DEFAULT


def _in_window(h: int, start: int, end: int) -> bool:
    """24h wrap-aware (e.g., 21..5 for NY)"""
    if start <= end:
        return start <= h <= end
    return h >= start or h <= end


def _session_from_hour(h: int) -> str:
    cfg = _load_cfg()
    for name, (a, b) in cfg.items():
        if _in_window(h, a, b):
            return name
    return "Other"


def add_session(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    if time_col not in df.columns:
        return df
    d = df.copy()
    t = pd.to_datetime(d[time_col], errors="coerce")
    d["session"] = t.dt.hour.map(lambda x: _session_from_hour(int(x)) if pd.notna(x) else "Other")
    return d
