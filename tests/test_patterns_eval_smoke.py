import json
from pathlib import Path

import pandas as pd

from scripts.eval_patterns import run as run_eval


def test_patterns_eval_smoke(tmp_path: Path):
    data_path = Path("data/USDJPY_15m.csv")
    assert data_path.exists(), "sample data/USDJPY_15m.csv is required"

    # Use a small slice for speed
    df = pd.read_csv(data_path).head(1500)
    local = tmp_path / "input.csv"
    df.to_csv(local, index=False)

    out_dir = tmp_path / "out"
    metrics_path, metrics = run_eval(
        data_path=str(local),
        out_dir=str(out_dir),
        H=4,
        delta_mult=0.5,
        spread_pips=0.5,
        pip_size=0.01,
    )

    # Files exist
    assert (out_dir / "signals.parquet").exists()
    assert (out_dir / "metrics.json").exists()

    # Basic keys
    with open(metrics_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    for k in [
        "H",
        "delta_mult",
        "spread_pips",
        "n_rows",
        "n_signals",
        "baseline_up",
        "baseline_down",
        "hit_rate",
        "ev_R",
        "verdict",
    ]:
        assert k in obj
