from __future__ import annotations

import pandas as pd
import streamlit as st

from typing import Optional

from features_util import augment_features
from ml.time_consistency import build_features


@st.cache_data(show_spinner=False)
def prepare_feats_cached(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cached wrapper around feature preparation used in inference.

    Accepts a minimal OHLCV DataFrame with 'timestamp' and returns the
    same schema as prepare_df_feats_for_inference in app.py.
    """
    # Normalize columns similar to app.prepare_df_feats_for_inference
    if not {"timestamp","open","high","low","close"}.issubset(set(c.lower() for c in raw_df.columns)):
        rename_map = {}
        for c in raw_df.columns:
            lc = c.lower()
            if lc in ["timestamp","open","high","low","close","volume"]:
                rename_map[c] = lc
        raw_df = raw_df.rename(columns=rename_map)

    raw_df = raw_df.copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df = raw_df.sort_values("timestamp").reset_index(drop=True)

    base_feats = build_features(raw_df)
    df_feats = augment_features(base_feats, raw_df.rename(columns=str.lower))
    df_feats = df_feats.fillna(0.0)
    if "timestamp" not in df_feats.columns:
        df_feats.insert(0, "timestamp", raw_df["timestamp"].values)
    return df_feats
