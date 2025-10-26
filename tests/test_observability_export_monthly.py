from datetime import datetime
from pathlib import Path

import pandas as pd

from src.monitoring.observability import export_monthly_report


def test_export_monthly_report(tmp_path: Path):
    df = pd.DataFrame(
        {
            "time": [datetime(2025, 1, 1, 9, 0), datetime(2025, 1, 1, 9, 1)],
            "trade_ok": [True, False],
            "reason": ["", "spread>1.0"],
        }
    )
    out = export_monthly_report(df, out_root=tmp_path, as_of=datetime(2025, 1, 31))
    assert out.exists()
    assert (out / "daily.csv").exists()
    assert (out / "top_reasons.csv").exists()

    meta = out / "metadata.json"
    assert meta.exists()
    txt = meta.read_text(encoding="utf-8")
    assert "generated_at" in txt and "artifacts" in txt
    # presence-only check (values may be null/empty)
    assert "git_commit" in txt and "config_hashes" in txt
