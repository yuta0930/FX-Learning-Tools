from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass
class EVConfig:
    R_win: float = 1.0
    R_loss: float = 1.0
    cost_per_trade: float = 0.15  # R換算（片道スプレッド＋スリッページ等を現実寄りに）

def ev_for_threshold(proba: np.ndarray, theta: float, ev: EVConfig) -> Dict[str, float]:
    sel = proba >= theta
    if sel.sum() == 0:
        return {"theta": float(theta), "trades": 0, "ev_per_trade": float("-inf"), "coverage": 0.0, "avg_p": 0.0}
    p = proba[sel]
    evp = (p * ev.R_win) - ((1.0 - p) * ev.R_loss) - ev.cost_per_trade
    return {
        "theta": float(theta),
        "trades": int(sel.sum()),
        "ev_per_trade": float(np.mean(evp)),
        "coverage": float(np.mean(sel)),
        "avg_p": float(np.mean(p)),
    }
