def save_calibration_json(cal, meta, out_json):
    import json
    d = {**meta, **cal.__dict__} if hasattr(cal, '__dict__') else {**meta, **cal}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from typing import List, Dict

from dataclasses import dataclass
# tools/calibration.py
import os, json, numpy as np, pandas as pd

# --- dataclass に CI と戦略を追加 ---
@dataclass
class CalibResult:
    n_bins: int
    bin_edges: List[float]
    prob_mean: List[float]
    frac_pos: List[float]
    counts: List[int]
    brier: float
    ece: float
    # NEW:
    frac_lo: List[float]
    frac_hi: List[float]
    strategy: str
    alpha: float

def _wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054  # ~ N(0,1) 97.5%（scipyなし）
    phat = k / n
    denom = 1.0 + (z**2)/n
    center = (phat + (z**2)/(2*n)) / denom
    half = (z * ((phat*(1-phat)/n + (z**2)/(4*n**2))**0.5)) / denom
    return max(0.0, center - half), min(1.0, center + half)

# 等頻度 or 等幅を選べるように
def _reliability_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10,
                       strategy: str = "quantile", alpha: float = 0.05) -> CalibResult:
    y = y_true.astype(int)
    p = proba.astype(float)
    assert p.min() >= 0 and p.max() <= 1, "proba must be in [0,1]"

    if strategy == "quantile":
        qs = np.linspace(0, 1, n_bins+1)
        bins = np.unique(np.quantile(p, qs))
        # p が単調でない/重複で bins が縮むときはフォールバック
        if len(bins) < 3:
            bins = np.linspace(0.0, 1.0, n_bins+1)
    else:
        bins = np.linspace(0.0, 1.0, n_bins+1)

    idx = np.digitize(p, bins, right=False) - 1
    idx = np.clip(idx, 0, len(bins)-2)

    prob_mean, frac_pos, counts, lo_list, hi_list = [], [], [], [], []
    for b in range(len(bins)-1):
        m = (idx == b)
        cnt = int(m.sum()); counts.append(cnt)
        if cnt == 0:
            prob_mean += [float("nan")]
            frac_pos += [float("nan")]
            lo_list   += [float("nan")]
            hi_list   += [float("nan")]
            continue
        prob_mean.append(float(p[m].mean()))
        pos = int(y[m].sum())
        freq = pos / cnt
        frac_pos.append(float(freq))
        lo, hi = _wilson_interval(pos, cnt, alpha=alpha)
        lo_list.append(float(lo)); hi_list.append(float(hi))

    brier = float(brier_score_loss(y, p))
    ece = 0.0
    for b in range(len(counts)):
        if counts[b] > 0 and not (np.isnan(prob_mean[b]) or np.isnan(frac_pos[b])):
            w = counts[b] / len(y)
            ece += w * abs(prob_mean[b] - frac_pos[b])

    return CalibResult(
        n_bins=len(bins)-1,
        bin_edges=list(map(float, bins)),
        prob_mean=prob_mean,
        frac_pos=frac_pos,
        counts=counts,
        brier=brier,
        ece=float(ece),
        frac_lo=lo_list,
        frac_hi=hi_list,
        strategy=strategy,
        alpha=alpha,
    )

def save_calibration_png(cal: CalibResult, title: str, out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    x = np.asarray(cal.prob_mean, float)
    y = np.asarray(cal.frac_pos, float)
    lo = np.asarray(cal.frac_lo, float)
    hi = np.asarray(cal.frac_hi, float)
    m = ~np.isnan(x) & ~np.isnan(y)
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1], linestyle="--", label="Ideal")
    if m.any():
        plt.plot(x[m], y[m], marker="o", label="Observed")
        m_ci = m & ~np.isnan(lo) & ~np.isnan(hi)
        if m_ci.any():
            # CI帯（連結して描く。点が離れる場合は見た目重視）
            plt.fill_between(x[m_ci], lo[m_ci], hi[m_ci], alpha=0.2, label="95% CI")
    plt.title(title + f"  ({cal.strategy}, bins={cal.n_bins})")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.xlim(0,1); plt.ylim(0,1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def calibration_report(y_true: np.ndarray, proba: np.ndarray, *,
                       n_bins: int = 10, title: str = "Calibration",
                       out_dir: str = "reports", stem: str = "break_calibration",
                       meta: Dict = None, strategy: str = "quantile", alpha: float = 0.05) -> Dict:
    cal = _reliability_table(y_true, proba, n_bins=n_bins, strategy=strategy, alpha=alpha)
    out_png = os.path.join(out_dir, f"{stem}.png")
    out_json= os.path.join(out_dir, f"{stem}.json")
    save_calibration_png(cal, title, out_png)
    save_calibration_json(cal, meta or {}, out_json)
    return {"png": out_png, "json": out_json, "brier": cal.brier, "ece": cal.ece, "counts": cal.counts}
    out_png = os.path.join(out_dir, f"{stem}.png")
    out_json= os.path.join(out_dir, f"{stem}.json")
    save_calibration_png(cal, title, out_png)
    save_calibration_json(cal, meta or {}, out_json)
    return {"png": out_png, "json": out_json, "brier": cal.brier, "ece": cal.ece, "counts": cal.counts}
