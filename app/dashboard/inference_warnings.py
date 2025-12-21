"""UI: inference-time warnings.

Purpose:
- Keep app.py thin by extracting small UI wrappers.
- Centralize how we show bars/day mismatch warnings (warn vs strict).

This module is Streamlit-specific by design.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from inference_break import _check_bars_per_day_consistency


def warn_bars_per_day_mismatch(df_feats: pd.DataFrame, meta: dict) -> None:
    """Show bars/day mismatch warning on Streamlit UI.

    - Non-strict: show st.warning (and inference_break will also print [warn])
    - Strict: show st.error and re-raise RuntimeError
    """

    try:
        warn_msg = _check_bars_per_day_consistency(df_feats, meta)
        if warn_msg:
            st.warning(warn_msg)
    except RuntimeError as e:
        st.error(str(e))
        raise
