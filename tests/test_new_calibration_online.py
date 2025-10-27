import numpy as np
import pandas as pd
from app.calibration.online_calibrator import OnlineCalibrator
from app.utils.stats import bin_ece


def test_online_calibrator_reduces_ece():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.5, size=1200)
    p_raw = y * 0.9 + (1 - y) * 0.1
    df = pd.DataFrame({"y": y, "p_raw": p_raw})
    ece_base, _ = bin_ece(y.astype(float), p_raw.astype(float), n_bins=10)
    cal = OnlineCalibrator(method="isotonic")
    cal.fit_and_eval(df)
    p_cal = cal.predict(p_raw)
    ece_cal, _ = bin_ece(y.astype(float), np.asarray(p_cal, float), n_bins=10)
    assert ece_cal <= ece_base * 0.7
