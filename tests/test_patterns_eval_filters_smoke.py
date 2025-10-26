from pathlib import Path
import subprocess, sys, json
import pandas as pd
from datetime import datetime, timedelta


def test_eval_patterns_filters_smoke(tmp_path: Path):
    # Minimal synthetic data
    csv = tmp_path / "m.csv"
    t0 = datetime(2025, 1, 1, 9, 0)
    n = 120
    base = [100.0 + 0.01 * i for i in range(n)]
    df = pd.DataFrame(
        {
            "time": [t0 + timedelta(minutes=15 * i) for i in range(n)],
            "open": base,
            "high": [b + 0.1 for b in base],
            "low": [b - 0.1 for b in base],
            "close": base,
            "spread_pips": [0.2] * n,
        }
    )
    df.to_csv(csv, index=False)

    # One event window roughly mid-series
    ev = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "time": [(t0 + timedelta(minutes=15 * 60)).strftime("%Y-%m-%d %H:%M")],
            "title": ["X"],
            "importance": ["high"],
        }
    ).to_csv(ev, index=False)

    # Run evaluation with filters applied
    out_root = tmp_path / "rep"
    cmd = [
        sys.executable,
        "scripts/eval_patterns.py",
        "--data",
        str(csv),
        "--out",
        str(out_root / "patterns_test"),
        "--max-spread-pips",
        "1.0",
        "--events-csv",
        str(ev),
        "--news-minutes-before",
        "30",
        "--news-minutes-after",
        "30",
        "--sessions-allow",
        "Tokyo,London,NewYork",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    # Check artifacts
    rep_dir = out_root / "patterns_test"
    assert rep_dir.exists(), "report dir missing"
    mfile = rep_dir / "metrics.json"
    assert mfile.exists()
    js = json.loads(mfile.read_text(encoding="utf-8"))
    assert "n_rows" in js and "n_signals" in js and "verdict" in js
