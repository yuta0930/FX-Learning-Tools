"""UI: data source selector panel.

Extracted from app.py to keep top-level script smaller and easier to maintain.
"""

from __future__ import annotations

import streamlit as st


def render_data_source_panel(
    *,
    key_prefix: str = "app_ds",
    default_path: str = "data/USDJPY_15m.csv",
) -> None:
    """Render optional data source chooser.

    Writes the selected path to st.session_state["default_data_path"].
    """

    with st.expander("データソース (任意) ⚙", expanded=False):
        cur_path = st.session_state.get("default_data_path", default_path)
        st.write(f"現在のパス: {cur_path}")
        new_path = st.text_input(
            "CSV/Parquetのパスを指定",
            value=str(cur_path),
            key=f"{key_prefix}_input_path",
        )
        c_ds1, c_ds2 = st.columns([1, 1])
        with c_ds1:
            if st.button("パスを適用", key=f"{key_prefix}_apply"):
                st.session_state["default_data_path"] = new_path
                st.success(f"データパスを更新しました: {new_path}")
        with c_ds2:
            if st.button("キャッシュクリア", key=f"{key_prefix}_clear_cache"):
                try:
                    st.cache_data.clear()
                    st.success("キャッシュをクリアしました。再描画してください。")
                except Exception as e:
                    st.info(f"キャッシュクリアに失敗: {e}")
