"""UI: leakage suspects report preview.

This is a small, low-risk extraction from the monolithic app.py.
It intentionally has no dependencies on the rest of the UI so that
app.py can stay thin and imports stay stable.
"""

from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
import streamlit as st


def render_leakage_suspects_report_sidebar(
    *,
    env_var: str = "LEAKAGE_REPORT_PATH",
    default_path: str = "reports/leakage_suspects.csv",
) -> None:
    """Render leakage suspects report section in Streamlit sidebar.

    If a CSV exists at the resolved path, show a preview and allow download.

    Resolution order:
      1) env_var (if set)
      2) default_path
    """

    rep_path = os.getenv(env_var, "").strip() or default_path
    p = Path(rep_path)

    with st.sidebar.expander("⚠️ Leakage suspects report", expanded=False):
        if p.exists() and p.is_file():
            st.warning(f"Leakage suspects report found: {rep_path}")
            try:
                rep = pd.read_csv(p)
                st.dataframe(rep.head(20), use_container_width=True)
                st.download_button(
                    label="Download report (CSV)",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Failed to load report: {e}")
        else:
            st.caption("No report found. Set LEAKAGE_REPORT_PATH to enable.")
