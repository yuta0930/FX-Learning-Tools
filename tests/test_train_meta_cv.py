import json
from pathlib import Path

import pandas as pd

from ai_train_break import save_meta
from ev_utils import EVConfig


def test_save_meta_records_cv_settings(tmp_path: Path):
    # Minimal DF satisfying save_meta contract
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="15min"),
            "y": [0, 1, 0],
        }
    )
    ev = EVConfig(R_win=1.0, R_loss=1.0, cost_per_trade=0.1)

    summary = {
        "AP_macro": 0.1,
        "Brier_macro": 0.2,
        "best_threshold": {"theta": 0.7, "coverage": 0.12, "ev_per_trade": 0.01},
        "use_cols": ["f1", "f2"],
        "oos_theta_eval": {"ev_per_trade": 0.01},
    }

    out_path = tmp_path / "meta.json"
    save_meta(
        df,
        ev,
        summary,
        str(out_path),
        n_splits=3,
        embargo_groups=2,
        group_gap=1,
    )

    meta = json.loads(out_path.read_text(encoding="utf-8"))
    assert meta["cv"]["kind"] == "PurgedGroupTimeSeriesSplit"
    assert meta["cv"]["n_splits"] == 3
    assert meta["cv"]["embargo_groups"] == 2
    assert meta["cv"]["group_gap"] == 1
