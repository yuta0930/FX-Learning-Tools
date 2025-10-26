from pathlib import Path
import subprocess, sys, json
import pandas as pd
from datetime import datetime, timedelta


def test_evcurve_smoke(tmp_path: Path):
    # Minimal data (~120 bars) to ensure quality/r_net columns are produced
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

    out_root = tmp_path / "rep"
    out_dir = out_root / "patterns_test"
    cmd = [
        sys.executable,
        "scripts/eval_patterns.py",
        "--data",
        str(csv),
        "--out",
        str(out_dir),
        "--evcurve-min-n",
        "0",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    # Check files
    assert out_dir.exists(), "report dir missing"
    evc = out_dir / "ev_curve.csv"
    assert evc.exists(), "ev_curve.csv not created"
    mfile = out_dir / "metrics.json"
    assert mfile.exists(), "metrics.json missing"
    js = json.loads(mfile.read_text(encoding="utf-8"))
    assert "ev_curve_best" in js, "ev_curve_best not present in metrics"
