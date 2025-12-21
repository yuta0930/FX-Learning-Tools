import streamlit as st

from app import _maybe_autoswitch_timeframe_profile


def test_switch_15m_to_1h_to_15m_reverts_defaults(monkeypatch):
    # Ensure a clean session_state for this test
    st.session_state.clear()

    # Start at 15m with default-like values
    st.session_state["interval"] = "15m"
    st.session_state["reg_lookback"] = 40
    st.session_state["dbscan_eps"] = 0.08
    st.session_state["zoom_recent_n"] = 100
    st.session_state["y_zoom_margin_pct"] = 0

    _maybe_autoswitch_timeframe_profile()

    # Go to 1h: should move to 1h defaults
    st.session_state["interval"] = "60m"
    _maybe_autoswitch_timeframe_profile()
    assert st.session_state["reg_lookback"] == 96
    assert abs(float(st.session_state["dbscan_eps"]) - 0.10) < 1e-12
    assert st.session_state["zoom_recent_n"] == 250
    assert int(st.session_state["y_zoom_margin_pct"]) == 0

    # Back to 15m: should revert to 15m defaults if still on 1h defaults
    st.session_state["interval"] = "15m"
    _maybe_autoswitch_timeframe_profile()
    assert st.session_state["reg_lookback"] == 40
    assert abs(float(st.session_state["dbscan_eps"]) - 0.08) < 1e-12
    assert st.session_state["zoom_recent_n"] == 100
    assert int(st.session_state["y_zoom_margin_pct"]) == 0


def test_back_to_15m_does_not_override_user_custom(monkeypatch):
    st.session_state.clear()

    # User uses an hourly chart but customizes values
    st.session_state["interval"] = "60m"
    st.session_state["reg_lookback"] = 123  # custom
    st.session_state["dbscan_eps"] = 0.123  # custom
    st.session_state["zoom_recent_n"] = 200  # custom
    st.session_state["y_zoom_margin_pct"] = 10  # custom

    _maybe_autoswitch_timeframe_profile()

    # Back to 15m should NOT override these custom values
    st.session_state["interval"] = "15m"
    _maybe_autoswitch_timeframe_profile()
    assert st.session_state["reg_lookback"] == 123
    assert abs(float(st.session_state["dbscan_eps"]) - 0.123) < 1e-12
    assert st.session_state["zoom_recent_n"] == 200
    assert int(st.session_state["y_zoom_margin_pct"]) == 10
