"""ATR utilities extracted from app.py.

Keep this module UI-independent to make it reusable in scripts/tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_latest_atr(price_df: pd.DataFrame, period: int = 14) -> float:
    """Compute latest ATR (Wilder EWM) from OHLC dataframe.

    The input can have case-insensitive columns; the function normalizes names.
    Returns NaN on invalid input.
    """

    try:
        req = {"high", "low", "close"}
        cols_lower = {c.lower() for c in price_df.columns}
        if not req.issubset(cols_lower):
            return float("nan")

        df = price_df.copy()
        ren: dict[str, str] = {}
        for c in df.columns:
            lc = c.lower()
            if lc in req:
                ren[c] = lc
        if ren:
            df = df.rename(columns=ren)

        # Drop incomplete last bar
        try:
            hh = pd.to_numeric(df["high"], errors="coerce").astype(float)
            ll = pd.to_numeric(df["low"], errors="coerce").astype(float)
            cc = pd.to_numeric(df["close"], errors="coerce").astype(float)
            mask = hh.notna() & ll.notna() & cc.notna()
            if len(mask) > 0 and not bool(mask.iloc[-1]):
                last_valid_idx = mask[::-1].idxmax()
                if pd.notna(last_valid_idx):
                    df = df.loc[:last_valid_idx]
        except Exception:
            pass

        hl = (df["high"] - df["low"]).abs()
        pc = df["close"].shift(1)
        h_pc = (df["high"] - pc).abs()
        l_pc = (df["low"] - pc).abs()
        tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1.0 / float(period), adjust=False).mean()
        if len(atr) == 0:
            return float("nan")

        val = atr.iloc[-1]
        if not np.isfinite(val):
            try:
                val = atr.dropna().iloc[-1]
            except Exception:
                return float("nan")

        return float(val) if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")
