from __future__ import annotations

from typing import Optional
import pandas as pd
from pydantic import BaseModel, Field

from src.core.ta import add_atr_if_missing, add_spread_pips_if_missing
from src.policy.filters import (
    apply_spread_filter,
    apply_atr_filter,
    apply_news_window_filter,
)
from src.policy.df_guard import enforce_env_guard_df
from src.policy.pattern_quality_gate import (
    enforce_pattern_quality_gate_df,
    load_thresholds_yaml,
)
from src.policy.reasons import unify_reasons_df


class FilterConfig(BaseModel):
    max_spread_pips: float = Field(default=1.0, ge=0.0)
    atr_min: Optional[float] = Field(default=None, ge=0.0)
    atr_max: Optional[float] = Field(default=2.0, ge=0.0)
    news_before_min: int = Field(default=10, ge=0)
    news_after_min: int = Field(default=10, ge=0)
    pip_size: float = Field(default=0.01, gt=0)


def run_market_filters(
    df: pd.DataFrame,
    *,
    events_df: Optional[pd.DataFrame] = None,
    cfg: Optional[FilterConfig] = None,
) -> pd.DataFrame:
    """Run a unified pipeline: safe column completion -> market filters -> env guard.

    - Adds spread_pips and atr only if missing and required inputs exist (safe no-op otherwise)
    - Applies spread / ATR / news-window filters (each a safe no-op if required columns are missing)
    - Applies environment guard (MODE/KILL) at the end
    """
    cfg = cfg or FilterConfig()
    out = df.copy()

    # Safe completion of missing columns (no effect if already present)
    out = add_spread_pips_if_missing(out, pip_size=cfg.pip_size)
    out = add_atr_if_missing(out)

    # Market filters (each is a no-op if columns are missing)
    out = apply_spread_filter(out, max_spread_pips=cfg.max_spread_pips)
    out = apply_atr_filter(out, min_atr=cfg.atr_min, max_atr=cfg.atr_max)
    if events_df is not None:
        out = apply_news_window_filter(
            out,
            events=events_df,
            minutes_before=cfg.news_before_min,
            minutes_after=cfg.news_after_min,
        )

    # Pattern quality gate (No-Op if no thresholds or no quality column)
    try:
        thr = load_thresholds_yaml()
        out = enforce_pattern_quality_gate_df(out, thresholds=thr)
    except Exception:
        # Do not break pipeline on gate failures
        pass

    # Final environment guard (MODE/KILL)
    out = enforce_env_guard_df(out)
    # Unify reasons into a single human-friendly column (optional, No-Op if not present)
    out = unify_reasons_df(out)
    return out
