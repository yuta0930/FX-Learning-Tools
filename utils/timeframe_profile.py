from __future__ import annotations


def minutes_from_interval(interval: str) -> int | None:
    """Convert interval string (e.g., '15m','60m','1d') to minutes if intraday."""
    try:
        s = str(interval).strip().lower()
        if s.endswith("m"):
            return int(s[:-1])
        if s.endswith("h"):
            return int(s[:-1]) * 60
        return None
    except Exception:
        return None


def bars_per_day(interval: str) -> int:
    """Bars per 24h day (used as fallback when timestamp is missing)."""
    m = minutes_from_interval(interval)
    if m and m > 0:
        return max(1, int(round(24 * 60 / m)))
    if str(interval).strip().lower() == "1d":
        return 1
    # Safe default: keep old behavior
    return 96


def recommended_refresh_secs(interval: str) -> int:
    """UI refresh default per timeframe (conservative)."""
    s = str(interval).strip().lower()
    if s in {"60m", "1h"}:
        return 600
    if s == "30m":
        return 300
    if s == "15m":
        return 180
    if s == "5m":
        return 60
    if s == "1d":
        return 600
    return 180


def _maybe_autoswitch_timeframe_profile() -> None:
    """Apply timeframe profile when interval changes.

    This is UI-facing behavior but is kept in a small module so it can be
    imported by tests without importing the full Streamlit app.
    """
    import streamlit as st

    try:
        interval = str(st.session_state.get("interval", "15m"))
    except Exception:
        interval = "15m"

    prev = st.session_state.get("_last_interval_for_profile")
    # Persist last interval if possible. In bare-mode (pytest), this write may not stick.
    st.session_state["_last_interval_for_profile"] = interval
    try:
        tracking_ok = st.session_state.get("_last_interval_for_profile") == interval
    except Exception:
        tracking_ok = True
    if prev == interval and tracking_ok:
        return

    st.session_state["bars_per_day"] = bars_per_day(interval)

    prev_norm = str(prev).strip().lower() if prev is not None else ""
    cur_norm = str(interval).strip().lower()

    # Defaults per profile
    defaults_15m = {
        "refresh_secs": recommended_refresh_secs("15m"),
        "reg_lookback": 40,
        "dbscan_eps": 0.08,
        "zoom_recent_n": 100,
        "y_zoom_margin_pct": 0,
        "default_data_path": "data/USDJPY_15m.csv",
    }
    defaults_60m = {
        "refresh_secs": recommended_refresh_secs("60m"),
        "reg_lookback": 96,
        "dbscan_eps": 0.10,
        "zoom_recent_n": 250,
        "y_zoom_margin_pct": 0,
        "default_data_path": "data/USDJPY_60m.csv",
    }

    def _profile_defaults(norm: str) -> dict:
        if norm in {"60m", "1h"}:
            return defaults_60m
        return defaults_15m

    prev_defs = _profile_defaults(prev_norm)
    cur_defs = _profile_defaults(cur_norm)

    # In bare-mode (e.g., pytest), Streamlit may warn that session_state doesn't function.
    # In that case, our internal tracking key can fail to persist, making `prev` unreliable.
    # To keep behavior stable, treat values equal to known defaults as "default-like" and allow switching.
    known_defaults = (defaults_15m, defaults_60m)

    def _set_if_default_or_missing(key: str) -> None:
        if key not in st.session_state:
            st.session_state[key] = cur_defs[key]
            return
        try:
            cur_val = st.session_state.get(key)

            # Normal case: update only if it was on the previous profile's default
            if prev is not None:
                if cur_val == prev_defs.get(key):
                    st.session_state[key] = cur_defs[key]
                return

            # Fallback: if we can't trust `prev`, update only when the value looks like a default
            for defs in known_defaults:
                if cur_val == defs.get(key):
                    st.session_state[key] = cur_defs[key]
                    break
        except Exception:
            pass

    # Refresh default: update only when it was on previous default
    _set_if_default_or_missing("refresh_secs")

    try:
        cur_path = str(st.session_state.get("default_data_path", "data/USDJPY_15m.csv"))
    except Exception:
        cur_path = "data/USDJPY_15m.csv"

    if interval in {"60m", "1h"}:
        # Switch data path only if it was the previous default
        if cur_path.replace("\\", "/") in {"data/usdjpy_15m.csv", "data/USDJPY_15m.csv"}:
            st.session_state["default_data_path"] = "data/USDJPY_60m.csv"
        _set_if_default_or_missing("reg_lookback")
        _set_if_default_or_missing("dbscan_eps")
        _set_if_default_or_missing("zoom_recent_n")
        _set_if_default_or_missing("y_zoom_margin_pct")

    elif interval == "15m":
        if cur_path.replace("\\", "/") in {"data/usdjpy_60m.csv", "data/USDJPY_60m.csv"}:
            st.session_state["default_data_path"] = "data/USDJPY_15m.csv"

        _set_if_default_or_missing("reg_lookback")
        _set_if_default_or_missing("dbscan_eps")
        _set_if_default_or_missing("zoom_recent_n")
        _set_if_default_or_missing("y_zoom_margin_pct")
