import os

import pandas as pd
import pytest


def test_inference_bars_per_day_warn_by_default(capsys, monkeypatch):
    from inference_break import _check_bars_per_day_consistency

    ts = pd.date_range("2025-01-01", periods=200, freq="1h", tz="UTC")  # 24 bars/day
    df = pd.DataFrame({"timestamp": ts})
    meta = {"bars_per_day_est": 96}  # trained on 15m

    monkeypatch.delenv("STRICT_BARS_PER_DAY", raising=False)
    msg = _check_bars_per_day_consistency(df, meta)
    out = capsys.readouterr().out
    assert "bars/day mismatch" in out
    assert isinstance(msg, str) and "bars/day mismatch" in msg


def test_inference_bars_per_day_strict_raises(monkeypatch):
    from inference_break import _check_bars_per_day_consistency

    ts = pd.date_range("2025-01-01", periods=200, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts})
    meta = {"bars_per_day_est": 96}

    monkeypatch.setenv("STRICT_BARS_PER_DAY", "1")
    with pytest.raises(RuntimeError):
        _check_bars_per_day_consistency(df, meta)
