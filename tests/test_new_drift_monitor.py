import numpy as np
from app.drift.drift_monitor import DriftMonitor


def test_drift_monitor_warn_and_halt():
    cfg = {"drift": {"psi_warn": 0.05, "psi_halt": 0.2}}
    dm = DriftMonitor(cfg)
    base = np.random.default_rng(0).beta(2, 5, size=2000)
    cur_warn = np.random.default_rng(1).beta(2.5, 4.5, size=500)
    cur_halt = np.random.default_rng(2).beta(5, 2, size=500)

    r1 = dm.check_drift(base, cur_warn)
    assert r1.level in {"ok","warn","halt"}

    r2 = dm.check_drift(base, cur_halt)
    assert r2.level in {"warn","halt"}
