from pathlib import Path
from datetime import datetime

import pandas as pd

from src.monitoring.health import assess_health, HealthThresholds


def test_assess_health_basic(tmp_path: Path):
    # 最小のログをCSVで用意
    df = pd.DataFrame({
        "time": [datetime.now(), datetime.now()],
        "trade_ok": [True, False],
        "reason": ["", "spread>1.0"],
    })
    p = tmp_path / "exec.csv"
    df.to_csv(p, index=False)

    th = HealthThresholds(lookback_days=1, max_block_rate=0.9, max_unknown_ratio=1.0)
    res = assess_health([p], th, None)
    assert res["ok"] is True


def test_assess_health_missing_logs_is_not_ok(tmp_path: Path):
    missing = tmp_path / "does_not_exist.parquet"
    th = HealthThresholds(lookback_days=1)
    res = assess_health([missing], th, None)
    assert res["ok"] is False
    assert res.get("error") == "no_log_files"
