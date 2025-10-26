import os
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd

# Ensure repo root is importable
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.eval_patterns_tri_rec_asia import run  # type: ignore


def make_rect_breakout_csv(path: Path, n: int = 400) -> Path:
    # 15m frequency
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    price = np.ones(n) * 150.0
    # create rectangle range [150-0.2, 150+0.2] for 150 bars
    rng = 0.2
    for i in range(50, 200):
        # alternate within range
        price[i] = 150.0 + (rng if (i % 2 == 0) else -rng)
    # breakout up
    for i in range(200, n):
        price[i] = price[i-1] + (0.03 if i < 260 else 0.0)
    df = pd.DataFrame({
        "time": idx,
        "open": price + np.random.normal(0, 0.01, n),
        "high": price + 0.02,
        "low": price - 0.02,
        "close": price,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def test_eval_smoke_outputs(tmp_path: Path):
    csv = make_rect_breakout_csv(tmp_path / "rect.csv")
    out_dir = tmp_path / "out"
    # Run evaluation with rectangle
    metrics_path, metrics = run(
        pattern="rectangle",
        data_path=str(csv),
        out_dir=str(out_dir),
        H=12,
        delta_mult=0.6,
        spread_pips=0.5,
        pip_size=0.01,
        max_spread_pips=None,
        events_csv=None,
        sessions_allow="London,NewYork",
        evcurve_min_n=10,
    )
    assert metrics_path.exists()
    # Required artifacts
    assert (out_dir / "signals.parquet").exists()
    assert (out_dir / "ev_curve.csv").exists()
    assert (out_dir / "session_summary.csv").exists()
    # Basic keys in metrics
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    for k in ["pattern", "n_signals", "hit_rate", "ev_R", "baseline_up", "baseline_down", "stage_counts"]:
        assert k in m
