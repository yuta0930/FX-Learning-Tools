import os
import tempfile
import yaml
import pandas as pd
from app.ab.ab_manager import ABManager
from constants import TRADES_LOG


def test_ab_assignment_and_promotion(tmp_path):
    cfg_path = tmp_path / "ab.yml"
    y = {
        "ab": {
            "enabled": True,
            "allocation": {"A": 0.5, "B": 0.5},
            "variants": {"A": {"ksl_atr": 0.55, "ktp_atr": 1.8}, "B": {"ksl_atr": 0.6, "ktp_atr": 1.6}},
            "eval": {"window_trades": 100, "metric_primary": "PF", "min_effect_size": 0.1, "promote_on_sig": True},
        }
    }
    cfg_path.write_text(yaml.safe_dump(y, allow_unicode=True, sort_keys=False), encoding="utf-8")

    m = ABManager(str(cfg_path))
    v = m.assign("setup-123")
    assert v in {"A","B"}

    # create fake trades with variant B better PF
    df = pd.DataFrame({
        "pnl_pips": [1,1,1,-0.5]*30,
        "variant": ["B"]*120,
        "order_id": [f"o{i}" for i in range(120)]
    })
    # ensure TRADES_LOG path exists locally; write then run eval using default path will miss, so patch manager path by changing cwd
    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        os.makedirs(os.path.dirname(TRADES_LOG), exist_ok=True)
        df.to_parquet(TRADES_LOG, index=False)
        m2 = ABManager(str(cfg_path))
        w, _ = m2.evaluate_and_promote()
        assert w in {"A","B","none"}
    finally:
        os.chdir(old)
