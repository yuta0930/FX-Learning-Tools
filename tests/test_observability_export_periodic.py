from pathlib import Path
from datetime import datetime, date, timedelta

import pandas as pd

from src.monitoring.observability import export_periodic_report


def test_export_periodic_report(tmp_path: Path):
    base = datetime(2025, 1, 1, 9, 0)
    df = pd.DataFrame(
        {
            "time": [base, base + timedelta(days=1), base + timedelta(days=2)],
            "trade_ok": [True, False, True],
            "reason": ["", "spread>1.0", ""],
        }
    )

    out = export_periodic_report(df, start=date(2025, 1, 1), end=date(2025, 1, 2), out_root=tmp_path)
    assert out.exists()
    assert (out / "daily.csv").exists()
    assert (out / "top_reasons.csv").exists()
    assert (out / "metadata.json").exists()
    meta = (out / "metadata.json").read_text(encoding="utf-8")
    # presence-only check (values may be null/empty)
    assert "git_commit" in meta and "config_hashes" in meta
