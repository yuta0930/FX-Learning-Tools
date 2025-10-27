from __future__ import annotations
import os
import time
import pandas as pd
import streamlit as st
import yaml
from constants import SIGNALS_LOG, ORDERS_LOG, TRADES_LOG
from app.utils.timeutil import JST

st.set_page_config(page_title="FX Monitoring Dashboard", page_icon="📊", layout="wide")

cfg = yaml.safe_load(open("configs/dashboard.yml", "r", encoding="utf-8"))
refresh = int(cfg.get("dashboard", {}).get("refresh_sec", 15))
st_autorefresh = st.empty()

st.title("Monitoring Dashboard")

sig = pd.read_parquet(SIGNALS_LOG) if os.path.exists(SIGNALS_LOG) else pd.DataFrame()
ord = pd.read_parquet(ORDERS_LOG) if os.path.exists(ORDERS_LOG) else pd.DataFrame()
trd = pd.read_parquet(TRADES_LOG) if os.path.exists(TRADES_LOG) else pd.DataFrame()

c1, c2, c3, c4 = st.columns(4)
if not trd.empty:
    W = (trd["pnl_pips"] > 0).mean()
    PF = trd.loc[trd["pnl_pips"] > 0, "pnl_pips"].sum() / max(-trd.loc[trd["pnl_pips"] <= 0, "pnl_pips"].sum(), 1e-6)
    EV = trd["pnl_pips"].mean()
    c1.metric("WinRate", f"{W:.2%}")
    c2.metric("PF", f"{PF:.2f}")
    c3.metric("EV/trade (pips)", f"{EV:.3f}")
    c4.metric("Trades", f"{len(trd)}")
else:
    c1.metric("WinRate", "-")
    c2.metric("PF", "-")
    c3.metric("EV/trade", "-")
    c4.metric("Trades", "0")

st.subheader("A/B variants")
if not trd.empty and "variant" in trd.columns:
    st.bar_chart(trd.groupby("variant")["pnl_pips"].mean())

st.subheader("Recent signals")
if not sig.empty:
    st.dataframe(sig.tail(20))

st.caption("Auto-refreshing")
st_autorefresh.caption(f"Refresh: {refresh}s")
