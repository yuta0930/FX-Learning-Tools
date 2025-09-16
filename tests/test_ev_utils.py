import numpy as np
from ev_utils import EVConfig, ev_for_threshold

def test_ev_for_threshold_no_trades():
    cfg = EVConfig()
    result = ev_for_threshold(np.array([0.1, 0.2]), 0.95, cfg)
    assert result["trades"] == 0
    assert np.isnan(result["ev_per_trade"])  # NaN expected
    assert result["coverage"] == 0.0


def test_ev_for_threshold_basic():
    cfg = EVConfig(R_win=1.0, R_loss=1.0, cost_per_trade=0.1)
    arr = np.array([0.55, 0.60, 0.90])
    out = ev_for_threshold(arr, 0.55, cfg)
    assert out["trades"] == 3
    assert 0.0 <= out["avg_p"] <= 1.0
    # Manual EV computation check
    sel = arr >= 0.55
    p = arr[sel]
    manual = ((p * cfg.R_win) - ((1 - p) * cfg.R_loss) - cfg.cost_per_trade).mean()
    assert abs(out["ev_per_trade"] - manual) < 1e-12
