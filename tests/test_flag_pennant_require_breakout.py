from __future__ import annotations

import numpy as np
import pandas as pd

from flag_pennant_detector import detect_flags_pennants


def _make_non_breakout(side: int = 1) -> pd.DataFrame:
    # Generate pole then flat consolidation WITHOUT actual breakout beyond buffer.
    pole_len = 25
    cons_len = 12
    base = 100.0
    pole_move = 0.3 * side
    pole = base + np.cumsum(np.full(pole_len, pole_move))
    cons_center = pole[-1]
    cons = cons_center + np.linspace(0, -0.2 * side, cons_len) + np.random.default_rng(0).normal(0, 0.02, cons_len)
    # final bar stays within channel -> no breakout
    final = cons[-1] + 0.05 * side  # still small, should not pass breakout_buffer
    close = np.concatenate([pole, cons, [final]])
    high = close + 0.03
    low = close - 0.03
    open_ = close.copy()
    idx = pd.date_range("2024-02-01", periods=len(close), freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_require_breakout_filters_patterns():
    df = _make_non_breakout(side=1)
    pats_no_req = detect_flags_pennants(df, pole_max_bars=30, pole_min_atr=1.5, cons_min_bars=6, cons_max_bars=16,
                                        require_breakout=False, quality_min=0.0)
    pats_req = detect_flags_pennants(df, pole_max_bars=30, pole_min_atr=1.5, cons_min_bars=6, cons_max_bars=16,
                                     require_breakout=True, quality_min=0.0)
    # If any pattern without requirement, with requirement should be <= and often 0.
    assert len(pats_req) <= len(pats_no_req)
    # Expect filtered out (synthetic lacks breakout amplitude).
    assert len(pats_req) == 0, "Breakout requirement failed to filter non-breakout pattern"
