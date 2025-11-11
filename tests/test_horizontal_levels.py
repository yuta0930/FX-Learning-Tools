import numpy as np
import pandas as pd

from utils.ta import horizontal_levels


def _df_from_vals(col, vals, index=None):
    if index is None:
        index = pd.RangeIndex(len(vals))
    return pd.DataFrame({col: vals}, index=index)


def test_horizontal_levels_empty():
    ph = pd.DataFrame()
    pl = pd.DataFrame()
    out = horizontal_levels(ph, pl, eps=None, min_samples=4)
    assert out == []


def test_horizontal_levels_merge_near():
    # Two close highs around 150.012 and 150.015 should merge with merge_near=0.01 (≈1 pip)
    highs = [150.012, 150.015, 151.00]
    ph = _df_from_vals("high", highs)
    pl = _df_from_vals("low", [])[:0]
    out = horizontal_levels(ph, pl, eps=None, min_samples=1, round_step=0.005, merge_near=0.01)
    # Expect 2 levels after merging closish two
    assert isinstance(out, list)
    assert len(out) == 2
    assert min(out) > 0


def test_horizontal_levels_rounding_pips():
    highs = [1.12345, 1.12346, 1.13001]
    lows  = [1.12340, 1.12347, 1.12999]
    ph = _df_from_vals("high", highs)
    pl = _df_from_vals("low", lows)
    out = horizontal_levels(ph, pl, eps=None, min_samples=2, round_step=0.00005, merge_near=0.0001)
    # Rounded to step, deduped
    assert len(out) >= 1
    # ensure sorted ascending
    assert out == sorted(out)
