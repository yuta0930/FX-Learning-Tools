from src.monitoring.metrics import brier_score, ece_score
import numpy as np

def test_metrics_smoke():
    y = np.array([0,1,0,1,1,0], dtype=float)
    p = np.array([0.2,0.7,0.4,0.8,0.6,0.3], dtype=float)
    b = brier_score(y,p)
    e = ece_score(y,p)
    assert 0.0 <= b <= 1.0
    assert 0.0 <= e <= 1.0
