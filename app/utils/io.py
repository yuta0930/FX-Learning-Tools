from __future__ import annotations
import os
from typing import Dict, Any, Optional, Union
import pandas as pd


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def append_parquet(df: Union[pd.DataFrame, Dict[str, Any]], path: str) -> None:
    """Append df to a parquet file, creating it if needed.
    Accepts a DataFrame or a single-row dict.
    """
    ensure_parent(path)
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    # Read-append in-memory, then atomic replace to reduce partial writes
    if os.path.exists(path):
        try:
            prev = pd.read_parquet(path)
            out = pd.concat([prev, df], ignore_index=True)
        except Exception:
            out = df.copy()
    else:
        out = df.copy()
    tmp_path = path + ".tmp-append"
    out.to_parquet(tmp_path, index=False)
    try:
        os.replace(tmp_path, path)
    except Exception:
        # Fallback: direct write if atomic replace fails
        out.to_parquet(path, index=False)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def read_parquet_safe(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        return None
    return None
