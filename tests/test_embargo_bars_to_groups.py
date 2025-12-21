import json

import pandas as pd


def test_embargo_groups_from_bars_15m_data():
    # 15m bars => 96 bars/day
    from ai_train_break import _embargo_groups_from_bars

    n = 96 * 4
    ts = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts})

    assert _embargo_groups_from_bars(df, 0) == 0
    assert _embargo_groups_from_bars(df, 1) == 1
    assert _embargo_groups_from_bars(df, 96) == 1
    assert _embargo_groups_from_bars(df, 97) == 2


def test_save_meta_persists_embargo_bars_and_label_cfg(tmp_path):
    from ai_train_break import save_meta

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC"),
            "y": [0, 1] * 5,
        }
    )
    meta_path = tmp_path / "meta.json"

    save_meta(
        df,
        None,
        {
            "embargo_bars": 97,
            "bars_per_day_est": 96,
            "label_cfg": {"horizon": 20, "buffer_ratio": 0.2},
        },
        str(meta_path),
        n_splits=3,
        embargo_groups=2,
        group_gap=1,
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["embargo_bars"] == 97
    assert meta["bars_per_day_est"] == 96
    assert meta["label_cfg"]["horizon"] == 20
    assert meta["cv"]["embargo_groups"] == 2
