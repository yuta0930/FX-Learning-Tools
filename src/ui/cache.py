from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import json


def file_mtime(path: str | Path) -> Optional[int]:
    try:
        p = Path(path)
        if not p.exists():
            return None
        # Use ns resolution to reduce false-equality on Windows FS
        return int(p.stat().st_mtime_ns)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_csv_cached(path: str | Path, mtime: Optional[float], dtype: Optional[Dict[str, Any]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    # Only pass simple args to keep the cache key stable
    return pd.read_csv(p, dtype=dtype, nrows=nrows)


@st.cache_data(show_spinner=False)
def read_parquet_cached(path: str | Path, mtime: Optional[float]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_parquet(p)


@st.cache_resource(show_spinner=False, ttl=600)
def joblib_load_cached(path: str | Path):
    from joblib import load  # local import

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return load(p)


@st.cache_data(show_spinner=False)
def read_json_cached(path: str | Path, mtime: Optional[float]):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))
