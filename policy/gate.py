"""Gate policy: decide trade_ok and reason from inputs.

This module is streamlit-free and testable. It receives a dict-like state
and returns a new DataFrame with trade_ok and optional reason/flags.
"""
from __future__ import annotations

from typing import Optional, Mapping
import pandas as pd
import numpy as np


def apply_final_gate(
    pred_df: pd.DataFrame,
    windows_df: Optional[pd.DataFrame],
    *,
    state: Mapping,
    signal_col: str = "signal",
    ts_col: str = "timestamp",
    add_columns: bool = True,
) -> pd.DataFrame:
    if not isinstance(pred_df, pd.DataFrame) or pred_df.empty:
        return pred_df
    df_out = pred_df.copy()

    # Read required flags from state, with defaults
    enable = bool(state.get("enable_trading", False))
    auto_pause = bool(state.get("auto_pause_on_drift", True))
    drift_state = state.get("drift_state", "normal")
    drift_block = auto_pause and (drift_state == "alert")

    # Guard (pass in via state['guard_state'])
    guard_ok = True
    gs = state.get("guard_state")
    if isinstance(gs, dict):
        guard_ok = not bool(gs.get("in_cooldown", False))

    # News hard suppression
    apply_news_filter = bool(state.get("apply_news_filter", False))
    news_block = False
    if (
        apply_news_filter
        and windows_df is not None
        and not windows_df.empty
        and ts_col in df_out.columns
    ):
        # Use a simple interval check; expect windows_df with ['start','end']
        ts = pd.to_datetime(df_out[ts_col])
        news_mask = pd.Series(False, index=df_out.index)
        try:
            starts = pd.to_datetime(windows_df["start"]).values
            ends = pd.to_datetime(windows_df["end"]).values
            for s, e in zip(starts, ends):
                news_mask |= (ts.values >= s) & (ts.values <= e)
        except Exception:
            news_mask = pd.Series(False, index=df_out.index)
        news_block = news_mask

    # Signal column
    sig = df_out[signal_col] if signal_col in df_out.columns else True
    sig_bool = sig.astype(bool) if hasattr(sig, "astype") else bool(sig)

    gate = enable and guard_ok and (not drift_block)
    news_ok = (~news_block) if isinstance(news_block, (pd.Series, np.ndarray)) else (not news_block)
    df_out["trade_ok"] = gate & sig_bool & news_ok

    if add_columns:
        # Reason column
        reason = pd.Series("ok", index=df_out.index, dtype=object)
        if not enable:
            reason = reason.mask(True, "disabled")
        if not guard_ok:
            reason = reason.mask(True, "guard_block")
        if drift_block:
            reason = reason.mask(True, "drift_block")
        if apply_news_filter:
            if isinstance(news_block, (pd.Series, np.ndarray)):
                reason = reason.mask(news_block, "news_block")
            elif news_block:
                reason = reason.mask(True, "news_block")
        if signal_col in df_out.columns:
            reason = reason.mask(~sig_bool, "no_signal")
        df_out["gate_reason"] = reason

        # Debug flags
        df_out["gate_enable"] = enable
        df_out["gate_guard_ok"] = guard_ok
        df_out["gate_drift_block"] = drift_block
        if apply_news_filter:
            df_out["gate_news_block"] = news_block if isinstance(news_block, (pd.Series, np.ndarray)) else bool(news_block)

    return df_out
