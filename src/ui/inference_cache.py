from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.ui.cache import joblib_load_cached, file_mtime


def _load_model_and_meta(model_path: str, meta_path: str):
    """Load model and meta with resource/data caches.

    Uses joblib_load_cached for model and a simple read for meta.
    """
    from inference_break import load_break_meta  # local import to avoid heavy globals

    model = joblib_load_cached(model_path)
    meta_mtime = file_mtime(meta_path)
    # cache meta by path+mtime
    meta = _read_meta_cached(meta_path, meta_mtime)
    return model, meta


@st.cache_data(show_spinner=False)
def _read_meta_cached(meta_path: str, mtime: Optional[float]):
    from inference_break import load_break_meta

    return load_break_meta(meta_path)


@st.cache_data(show_spinner=False)
def predict_cached(
    df_feats: pd.DataFrame,
    model_path: str,
    meta_path: str,
    use_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Cached prediction using df_feats and on-disk model/meta with mtime invalidation.

    - Hashes df_feats content; invalidates when model/meta mtime changes
    - Determines effective use_cols from override, meta["features"], or model defaults
    """
    from inference_break import load_break_model, predict_with_session_theta

    # Load model (resource-cached) and meta (data-cached by mtime)
    model, meta = _load_model_and_meta(model_path, meta_path)
    # Fallback to model's Xcols via load_break_model helper
    # load_break_model returns (model, Xcols) in this project
    _, Xcols = load_break_model(model_path)
    eff_cols: List[str] = (use_cols or (meta.get("features") if isinstance(meta, dict) else None) or Xcols)  # type: ignore
    return predict_with_session_theta(df_feats, model, eff_cols, meta)
