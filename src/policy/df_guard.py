from __future__ import annotations

import pandas as pd
from src.core.safety import env_guard_active


def _append_reason(col: pd.Series, reason: str) -> pd.Series:
    """Append a textual reason to an existing Series of messages (pipe-join)."""
    def add(msg):
        if pd.isna(msg) or msg is None or msg == "":
            return reason
        return f"{msg} | {reason}"

    return col.astype("object").map(add)


def enforce_env_guard_df(df: pd.DataFrame, reason: str = "disabled by MODE/KILL_SWITCH") -> pd.DataFrame:
    """
    If trading_enabled() is False:
      - Ensure df['trade_ok'] is present and set to False (or AND False if present)
      - Append reason into df['deny_reason'] (create if missing)
    Otherwise return df unchanged.
    """
    if not env_guard_active():
        return df

    out = df.copy()
    if "trade_ok" in out.columns:
        out["trade_ok"] = out["trade_ok"].astype(bool) & False
    else:
        out["trade_ok"] = False

    if "deny_reason" in out.columns:
        out["deny_reason"] = _append_reason(out["deny_reason"], reason)
    else:
        # broadcast single reason; pandas will align as scalar assignment
        out["deny_reason"] = reason

    return out
