import numpy as np
import pandas as pd
from src.core.drift import window_drift


def test_window_drift_sanity():
    # 同一分布: 乱数で2分割
    rng = np.random.RandomState(0)
    s_same = pd.Series(rng.randn(300))
    m_same = window_drift(s_same, ref_n=150, cur_n=150, bins=10)

    # シフトした分布: 後半を+3.0シフト
    rng2 = np.random.RandomState(0)
    ref_part = rng2.randn(150)
    cur_part = rng2.randn(150) + 3.0
    s_shift = pd.Series(np.concatenate([ref_part, cur_part]))
    m_shift = window_drift(s_shift, ref_n=150, cur_n=150, bins=10)

    for k in ("psi", "js"):
        assert np.isfinite(m_same[k]) and m_same[k] >= 0
        assert np.isfinite(m_shift[k]) and m_shift[k] >= 0

    # 同一分布はシフトより小さい（緩い判定）
    assert m_same["psi"] < m_shift["psi"]
    assert m_same["js"] <= m_shift["js"]
