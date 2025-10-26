from __future__ import annotations

import pandas as pd
import numpy as np


def add_atr_if_missing(
    df: pd.DataFrame,
    *,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    out_col: str = "atr",
) -> pd.DataFrame:
    """Add ATR column if it's missing and OHLC columns exist.

    Uses Wilder's moving average approximation via EWM.
    Safe no-op if requirements are not met.
    """
    if out_col in df.columns:
        return df

    required = {high_col, low_col, close_col}
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    h = pd.to_numeric(out[high_col], errors="coerce").astype(float)
    l = pd.to_numeric(out[low_col], errors="coerce").astype(float)
    c = pd.to_numeric(out[close_col], errors="coerce").astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    # Wilder: use EWM with alpha = 1/period
    atr = tr.ewm(alpha=1.0 / float(period), adjust=False).mean()
    out[out_col] = atr
    return out


def add_spread_pips_if_missing(
    df: pd.DataFrame,
    *,
    bid_col: str = "bid",
    ask_col: str = "ask",
    out_col: str = "spread_pips",
    pip_size: float = 0.01,  # USDJPY pairs typically use 0.01 pip size
) -> pd.DataFrame:
    """Add spread in pips if bid/ask exist and output column is missing.

    Safe no-op if requirements are not met.
    """
    if out_col in df.columns:
        return df
    if bid_col not in df.columns or ask_col not in df.columns:
        return df

    out = df.copy()
    bid = pd.to_numeric(out[bid_col], errors="coerce").astype(float)
    ask = pd.to_numeric(out[ask_col], errors="coerce").astype(float)
    out[out_col] = (ask - bid) / float(pip_size)
    return out
