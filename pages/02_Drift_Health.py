import streamlit as st
from pathlib import Path
import pandas as pd
from typing import Optional

st.set_page_config(page_title="Drift & Health", page_icon="🩺", layout="wide")
st.title("Drift & Health")

DATA_PATH_DEFAULT = Path("data/USDJPY_15m.csv")

with st.sidebar:
    st.subheader("Data Source")
    data_path_str = st.text_input("CSV path (OHLCV想定)", str(DATA_PATH_DEFAULT))
    ref_n = st.number_input("Reference window (rows)", min_value=500, value=2000, step=100)
    cur_n = st.number_input("Current window (rows)", min_value=100, value=500, step=50)
    bins = st.number_input("Histogram bins", min_value=5, value=20, step=1)


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None


df = load_csv(Path(data_path_str))
if df is None:
    st.info("CSV が見つかりません。data/USDJPY_15m.csv などを配置してください（列: time, open, high, low, close, volume を想定）。")
    st.stop()

if "close" not in df.columns:
    st.error("CSV に 'close' 列が必要です。")
    st.stop()

st.write("Rows:", len(df))

from src.core.drift import window_drift

try:
    metrics = window_drift(df["close"], ref_n=int(ref_n), cur_n=int(cur_n), bins=int(bins))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PSI", f"{metrics['psi']:.4f}")
    c2.metric("JS Divergence", f"{metrics['js']:.4f}")
    c3.metric("Ref size", metrics["ref_size"])
    c4.metric("Cur size", metrics["cur_size"])
    st.caption("PSI≈0.1超: 軽度ドリフト、0.25超: 強いドリフトの目安（一般論）")
except Exception as e:
    st.error(f"Drift計算に失敗しました: {e}")

with st.expander("Plot"):
    try:
        st.line_chart(df["close"].tail(int(ref_n) + int(cur_n)))
    except Exception:
        pass
