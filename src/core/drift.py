from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
import pandas as pd

EPS = 1e-12


def _hist_probs(x: pd.Series, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Return histogram probabilities and edges for a cleaned series."""
    xx = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    hist, bin_edges = np.histogram(xx.values, bins=bins)
    p = hist.astype(float)
    denom = max(p.sum(), 1.0)
    p = p / denom
    p = np.clip(p, EPS, 1.0)
    return p, bin_edges


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between ref and cur using shared bins.

    psi = sum((p - q) * log(p/q)), where p and q are binned frequencies.
    """
    xref = pd.to_numeric(ref, errors="coerce")
    xcur = pd.to_numeric(cur, errors="coerce")
    x = pd.concat([xref, xcur]).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return float("nan")
    hist_ref, edges = np.histogram(xref.replace([np.inf, -np.inf], np.nan).dropna().values, bins=np.histogram_bin_edges(x.values, bins=bins))
    hist_cur, _ = np.histogram(xcur.replace([np.inf, -np.inf], np.nan).dropna().values, bins=edges)
    p = np.clip(hist_ref.astype(float) / max(hist_ref.sum(), 1.0), EPS, 1.0)
    q = np.clip(hist_cur.astype(float) / max(hist_cur.sum(), 1.0), EPS, 1.0)
    return float(((p - q) * np.log(p / q)).sum())


def js_divergence(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Jensen–Shannon divergence between ref and cur (symmetric, bounded)."""
    xref = pd.to_numeric(ref, errors="coerce")
    xcur = pd.to_numeric(cur, errors="coerce")
    x = pd.concat([xref, xcur]).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return float("nan")
    hist_ref, edges = np.histogram(xref.replace([np.inf, -np.inf], np.nan).dropna().values, bins=np.histogram_bin_edges(x.values, bins=bins))
    hist_cur, _ = np.histogram(xcur.replace([np.inf, -np.inf], np.nan).dropna().values, bins=edges)
    p = np.clip(hist_ref.astype(float) / max(hist_ref.sum(), 1.0), EPS, 1.0)
    q = np.clip(hist_cur.astype(float) / max(hist_cur.sum(), 1.0), EPS, 1.0)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return float(0.5 * (kl_pm + kl_qm))


def window_drift(series: pd.Series, ref_n: int, cur_n: int, bins: int = 10) -> Dict[str, float]:
    """Compute drift metrics using the last cur_n samples as current and the
    preceding ref_n samples as reference.
    """
    s = pd.to_numeric(series, errors="coerce")
    if len(s) < ref_n + cur_n:
        raise ValueError("series length is too short for selected windows")
    cur = s.iloc[-cur_n:]
    ref = s.iloc[-(ref_n + cur_n) : -cur_n]
    return {
        "psi": psi(ref, cur, bins=bins),
        "js": js_divergence(ref, cur, bins=bins),
        "ref_size": int(ref_n),
        "cur_size": int(cur_n),
        "bins": int(bins),
    }
