from __future__ import annotations
import numpy as np
from typing import Dict, Tuple


def bin_ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict]:
    """Expected Calibration Error with equal-frequency bins.
    Returns (ece, details)
    """
    y = y_true.astype(float)
    p = p.astype(float)
    order = np.argsort(p)
    y = y[order]
    p = p[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    abs_errs = []
    bins_out = []
    for idx in bins:
        if len(idx) == 0:
            continue
        py = y[idx].mean()
        pp = p[idx].mean()
        w = len(idx) / len(p)
        abs_errs.append(w * abs(py - pp))
        bins_out.append({"count": int(len(idx)), "y": float(py), "p": float(pp)})
    ece = float(np.sum(abs_errs)) if abs_errs else float("nan")
    return ece, {"bins": bins_out}
