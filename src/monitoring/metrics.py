import numpy as np


def brier_score(y_true, p_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    m = ~np.isnan(y)
    y = y[m]; p = p[m]
    return float(np.mean((p - y) ** 2))


def ece_score(y_true, p_pred, n_bins: int = 15):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    m = ~np.isnan(y)
    y = y[m]; p = p[m]
    eps = 1e-12
    bins = np.linspace(0.0, 1.0 + eps, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)
