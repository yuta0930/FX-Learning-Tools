import numpy as np
import pandas as pd

from ai_train_break import make_dataset
from label_break import BreakLabelConfig


def _raw_df(n: int = 300) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="15min")
    base = 150.0
    # simple random walk-ish, deterministic
    close = base + np.cumsum(np.sin(np.arange(n) / 10.0) * 0.01)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.02
    low = np.minimum(open_, close) - 0.02
    vol = np.full(n, 100.0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


def test_dir_features_disabled_by_default():
    raw = _raw_df(260)
    cfg = BreakLabelConfig(H=12, buffer_mode="atr", break_buffer_atr=0.15, exclude_in_windows=False)
    df = make_dataset(raw, horizon_bars=12, buffer_ratio=0.15, label_config=cfg)
    assert "dir" not in df.columns
    assert "dir_sign" not in df.columns
    assert "dist_to_level" not in df.columns


def test_dir_features_enabled_with_flag():
    raw = _raw_df(260)
    cfg = BreakLabelConfig(H=12, buffer_mode="atr", break_buffer_atr=0.15, exclude_in_windows=False)
    df = make_dataset(
        raw,
        horizon_bars=12,
        buffer_ratio=0.15,
        label_config=cfg,
        enable_legacy_dir_features=True,
    )
    assert "dir" in df.columns
    assert set(df["dir"].unique()).issubset({-1, 1})
