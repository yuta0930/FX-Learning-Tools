import streamlit as st
from pathlib import Path
from src.ui.cache import file_mtime, read_json_cached

st.set_page_config(page_title="Backtest Summary", page_icon="📈", layout="wide")
st.title("Backtest Summary")

reports_dir = Path("reports")
wf_json = reports_dir / "wf_report.json"
equity_png = reports_dir / "equity_wf.png"
calib_json = reports_dir / "break_calibration.json"

st.subheader("Artifacts")
cols = st.columns(3)
cols[0].write(f"wf_report.json: {'✅' if wf_json.exists() else '❌'}")
cols[1].write(f"equity_wf.png: {'✅' if equity_png.exists() else '❌'}")
cols[2].write(f"break_calibration.json: {'✅' if calib_json.exists() else '❌'}")

if equity_png.exists():
    st.image(str(equity_png), caption="Walk-Forward Equity")

if wf_json.exists():
    try:
        m = file_mtime(wf_json)
        data = read_json_cached(str(wf_json), m)
        st.subheader("WF Report (excerpt)")
        st.json(data if isinstance(data, dict) else {"wf_report": data})
    except Exception as e:
        st.error(f"wf_report.json の読み込みに失敗: {e}")

st.caption("※ まだ自動実行は行いません。scripts/run_backtest.py を個別に実行して成果物を配置してください。")
