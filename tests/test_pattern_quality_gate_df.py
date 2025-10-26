from pathlib import Path
import pandas as pd

from src.policy.pattern_quality_gate import enforce_pattern_quality_gate_df


def test_quality_gate_by_session_and_default():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01 09:00", periods=5, freq="15min"),
            "session": ["Tokyo", "Tokyo", "London", "NewYork", "Other"],
            "pattern_quality": [0.55, 0.65, 0.57, 0.61, 0.54],
            "trade_ok": [True] * 5,
        }
    )

    thr = {"Tokyo": 0.60, "London": 0.58, "default": 0.56}
    out = enforce_pattern_quality_gate_df(df, thresholds=thr)

    # Tokyo 0.55 (<0.60) is blocked; 0.65 remains
    assert out.loc[0, "trade_ok"] is False
    assert out.loc[1, "trade_ok"] is True

    # London 0.57 (<0.58) is blocked
    assert out.loc[2, "trade_ok"] is False

    # NewYork uses default=0.56; 0.61 remains
    assert out.loc[3, "trade_ok"] is True

    # Other uses default=0.56; 0.54 blocked
    assert out.loc[4, "trade_ok"] is False

    # Reason present and contains marker
    assert "deny_reason" in out.columns
    assert "pattern_quality<thr" in str(out.loc[0, "deny_reason"]) or str(out.loc[0, "deny_reason"]) == "pattern_quality<thr"


essentially_true = [True, True, True]

def test_quality_gate_noop_when_no_quality_col():
    df = pd.DataFrame({"time": [1, 2, 3], "session": ["Tokyo"] * 3, "trade_ok": [True] * 3})
    out = enforce_pattern_quality_gate_df(df, thresholds={"default": 0.6})
    # No quality column -> No-Op
    assert (out["trade_ok"].tolist() == [True, True, True])
