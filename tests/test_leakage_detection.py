import pandas as pd
import numpy as np
from leakage_detection import detect_temporal_leakage


def test_detect_temporal_leakage_flags_future_like():
    n = 1200
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(123)
    # base target with slight imbalance
    y = (rng.random(n) < 0.18).astype(int)
    # noise feature
    noise = rng.normal(size=n)
    # future-leaky feature: future y with slight noise
    future_like = np.roll(y, -1).astype(float)
    future_like[-1] = y[-1]
    future_like += rng.normal(scale=0.02, size=n)

    df = pd.DataFrame({
        "timestamp": ts,
        "y": y,
        "noise": noise,
        "future_like": future_like,
    })

    rep = detect_temporal_leakage(df, min_samples=500)
    # future_like should appear and likely suspicious
    row = rep.loc[rep["feature"] == "future_like"].iloc[0]
    assert row["suspicious"], f"future_like expected suspicious but got {row.to_dict()}"


def test_detect_temporal_leakage_no_false_positive_on_noise():
    n = 1000
    ts = pd.date_range("2024-02-01", periods=n, freq="15min")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame({
        "timestamp": ts,
        "y": y,
        "noise1": rng.normal(size=n),
        "noise2": rng.normal(size=n),
    })
    rep = detect_temporal_leakage(df, min_samples=400)
    suspicious = rep.loc[rep['suspicious']]
    # allow at most 1 random false positive due to statistical fluctuation
    assert len(suspicious) <= 1, f"Too many false positives: {suspicious}"
