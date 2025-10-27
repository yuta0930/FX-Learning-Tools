import streamlit as st
from pathlib import Path

try:
    from src.core.safety import current_mode, is_kill_switch_on, trading_enabled, ensure_flags_dir, KILL_FILE
except Exception:
    # Fallbacks if imports fail on first run
    current_mode = lambda: "unknown"  # type: ignore
    is_kill_switch_on = lambda: True  # type: ignore
    trading_enabled = lambda: False  # type: ignore
    def ensure_flags_dir():
        pass
    KILL_FILE = Path("flags/kill.switch")  # type: ignore

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("Dashboard")
try:
    from src.ui.safety_badge import render_safety_badge
    render_safety_badge()
except Exception:
    pass

# ATR/kSL/kTP panel (always-on, safe no-op)
try:
    from src.ui.atr_panel import render_atr_panel
    # Data source selector (optional)
    with st.expander("データソース (任意) ⚙", expanded=False):
        cur_path = st.session_state.get("default_data_path", "data/USDJPY_15m.csv")
        st.write(f"現在のパス: {cur_path}")
        new_path = st.text_input("CSV/Parquetのパスを指定", value=str(cur_path), key="ds_input_path")
        cols = st.columns([1,1,2])
        with cols[0]:
            if st.button("パスを適用"):
                st.session_state["default_data_path"] = new_path
                st.success(f"データパスを更新しました: {new_path}")
        with cols[1]:
            if st.button("キャッシュクリア"):
                try:
                    st.cache_data.clear()
                    st.success("キャッシュをクリアしました。再描画してください。")
                except Exception as e:
                    st.info(f"キャッシュクリアに失敗: {e}")

    render_atr_panel(data_path=st.session_state.get("default_data_path", "data/USDJPY_15m.csv"))
except Exception:
    pass

ensure_flags_dir()

col1, col2, col3 = st.columns(3)
col1.metric("MODE", current_mode())
col2.metric("Kill Switch", "ON" if is_kill_switch_on() else "OFF")
col3.metric("Trading Enabled", "YES" if trading_enabled() else "NO")

st.caption("Status comes from environment variables and 'flags/kill.switch'.")

st.subheader("Kill Switch (file) control")
c_on, c_off = st.columns(2)
if c_on.button("🛑 Enable Kill Switch (file)", use_container_width=True):
    try:
        KILL_FILE.parent.mkdir(parents=True, exist_ok=True)
        KILL_FILE.write_text("on")
        st.success(f"Kill switch file created: {KILL_FILE}")
    except Exception as e:
        st.error(f"Failed to create kill switch file: {e}")

if c_off.button("✅ Disable Kill Switch (file)", use_container_width=True):
    try:
        if KILL_FILE.exists():
            KILL_FILE.unlink()
        st.success("Kill switch file removed.")
    except Exception as e:
        st.error(f"Failed to remove kill switch file: {e}")

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from src.policy.reasons import unify_reasons_df

st.subheader("Recent Blocks (optional)")
logs_path = Path("logs/executions.parquet")
if logs_path.exists():
    try:
        df_logs = pd.read_parquet(logs_path)
        # 直近7日だけ（time列があれば）
        if "time" in df_logs.columns:
            try:
                df_logs["time"] = pd.to_datetime(df_logs["time"], errors="coerce")
                since = datetime.now() - timedelta(days=7)
                df_logs = df_logs[df_logs["time"] >= since]
            except Exception:
                pass
        df_logs = unify_reasons_df(df_logs)
        # Blockだけに絞る（列があれば）
        if "trade_ok" in df_logs.columns:
            df_logs = df_logs[df_logs["trade_ok"] == False]  # noqa: E712
        reason_col = next((c for c in ["reason", "deny_reason", "gate_reason"] if c in df_logs.columns), None)
        if reason_col and not df_logs.empty:
            vc = (
                df_logs[reason_col]
                .astype(str)
                .replace({"": None, "nan": None})
                .dropna()
                .value_counts()
                .head(10)
            )
            st.write(vc)
            try:
                st.bar_chart(vc)
            except Exception:
                pass
        else:
            st.info("直近7日で Block の記録が見つかりませんでした。")
    except Exception as e:
        st.info(f"logs/executions.parquet の読み込みに失敗（省略します）: {e}")
else:
    st.caption("logs/executions.parquet があれば、直近のブロック理由がここに出ます。")
