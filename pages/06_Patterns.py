from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
from src.ui.cache import file_mtime, read_csv_cached, read_parquet_cached, read_json_cached

st.set_page_config(page_title="Patterns (Triangle/Rectangle/Asia)", page_icon="📐", layout="wide")
st.title("Patterns: Triangle / Rectangle / Asia Box")
try:
    from src.ui.safety_badge import render_safety_badge
    render_safety_badge()
except Exception:
    pass

# ATR/kSL/kTP panel (always-on, safe no-op)
try:
    from src.ui.atr_panel import render_atr_panel
    render_atr_panel(data_path=st.session_state.get("default_data_path", "data/USDJPY_15m.csv"))
except Exception:
    pass

st.caption("このページは新しいパターン検出の可視化とデバッグ補助用です。データが無い場合も落ちません。")

pat = st.sidebar.selectbox("Pattern", ["triangle", "rectangle", "asia_box"])  # noqa: F821
root = Path("reports")

# Find latest patterns_YYYYMMDD directory
candidates = sorted([p for p in root.glob("patterns_*") if p.is_dir()])
latest = candidates[-1] if candidates else None

if latest is None:
    st.info("まだレポートがありません。scripts/eval_patterns_tri_rec_asia.py で生成できます。")
    st.stop()

st.write(f"Latest directory: {latest}")

mfile = latest / "metrics.json"
evfile = latest / "ev_curve.csv"
sigfile = latest / "signals.parquet"

if not mfile.exists():
    st.info("metrics.json が見つかりませんでした。評価スクリプトを実行してください。")
    st.stop()

try:
    mm = file_mtime(mfile)
    metrics = read_json_cached(str(mfile), mm)
except Exception as e:
    st.info(f"metrics.json の読込に失敗: {e}")
    st.stop()

# Stage counters
st.subheader("件数サマリ")
st.json(metrics.get("stage_counts", {}))

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("n_signals", f"{metrics.get('n_signals', 0)}")
with c2:
    st.metric("hit_rate", f"{metrics.get('hit_rate', float('nan')):.3f}" if metrics.get('hit_rate') is not None else "-")
with c3:
    st.metric("EV_net(R)", f"{metrics.get('ev_R', float('nan')):.3f}" if metrics.get('ev_R') is not None else "-")

st.subheader("EV curve (by quality)")
if evfile.exists():
    try:
        em = file_mtime(evfile)
        df_ev = read_csv_cached(str(evfile), em)
        st.dataframe(df_ev.tail(10), use_container_width=True)
    except Exception:
        st.caption("ev_curve.csv は表示できませんでした。")
else:
    st.caption("ev_curve.csv は未生成です。")

st.subheader("直近サンプル（最大5件）")
if sigfile.exists():
    try:
        sm = file_mtime(sigfile)
        sdf = read_parquet_cached(str(sigfile), sm)
        if len(sdf) == 0:
            st.caption("signals.parquet は空です。")
        else:
            st.dataframe(sdf.tail(5), use_container_width=True)
    except Exception:
        st.caption("signals.parquet の読込に失敗しました。")
else:
    st.caption("signals.parquet は未生成です。")
