from __future__ import annotations

import os
import streamlit as st

from src.core.safety import trading_enabled, KILL_FILE, ensure_flags_dir


def render_safety_badge() -> None:
    """MODE / KILL の状況をページ上部に一貫表示。"""
    try:
        ensure_flags_dir()
    except Exception:
        pass

    mode = os.getenv("MODE", "unknown").lower()
    kill_env = os.getenv("KILL_SWITCH", "")
    kill_file = False
    try:
        kill_file = KILL_FILE.exists()
    except Exception:
        kill_file = False
    kill_on = (kill_env == "1") or kill_file

    enabled = False
    try:
        enabled = trading_enabled()
    except Exception:
        enabled = False

    if enabled:
        st.success(f"✅ Trading ENABLED | MODE={mode} | KILL={'off'}")
    else:
        st.error(
            f"⛔ Trading DISABLED | MODE={mode} | KILL={'on' if kill_on else 'off'}（env/file いずれか）"
        )
