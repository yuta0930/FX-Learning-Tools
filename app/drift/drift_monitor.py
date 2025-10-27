from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from monitoring import compute_psi
from app.utils.stats import bin_ece


@dataclass
class DriftCheckResult:
    level: str  # "ok"|"warn"|"halt"
    adjust: Dict
    metrics: Dict


class DriftMonitor:
    """Compute PSI/ECE from recent vs baseline data and suggest gate adjustment."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg.get("drift", {}) if cfg else {}

    def check_drift(self, baseline_probs: np.ndarray, current_probs: np.ndarray,
                    y_recent: Optional[np.ndarray] = None) -> DriftCheckResult:
        if baseline_probs is None or current_probs is None or len(baseline_probs) < 50 or len(current_probs) < 50:
            return DriftCheckResult(level="ok", adjust={}, metrics={})
        psi, _ = compute_psi(np.asarray(baseline_probs, float), np.asarray(current_probs, float))
        ece = None
        if y_recent is not None and len(y_recent) == len(current_probs):
            ece, _ = bin_ece(y_recent.astype(float), np.asarray(current_probs, float), n_bins=10)
        psi_warn = float(self.cfg.get("psi_warn", 0.25))
        psi_halt = float(self.cfg.get("psi_halt", 0.50))
        ece_warn = float(self.cfg.get("ece_warn", 0.03))
        ece_halt = float(self.cfg.get("ece_halt", 0.06))
        level = "ok"
        if (psi is not None and psi >= psi_halt) or (ece is not None and ece >= ece_halt):
            level = "halt"
        elif (psi is not None and psi >= psi_warn) or (ece is not None and ece >= ece_warn):
            level = "warn"
        adjust = {}
        actions = (self.cfg.get("actions") or {})
        if level == "warn":
            adjust = actions.get("warn", {"quality_add": 0.02, "p_add": 0.01})
        elif level == "halt":
            adjust = actions.get("halt", {"entry_stop": True})
        return DriftCheckResult(level=level, adjust=adjust, metrics={"psi": psi, "ece": ece})

    @staticmethod
    def apply_gate_adjustment(level: str, gate_cfg: Dict, adjust: Dict) -> Dict:
        g = dict(gate_cfg)
        if level == "warn":
            g["min_quality"] = float(g.get("min_quality", 0.75)) + float(adjust.get("quality_add", 0.0))
            g["p_threshold_break"] = float(g.get("p_threshold_break", 0.56)) + float(adjust.get("p_add", 0.0))
        elif level == "halt":
            g["entry_stop"] = True
        return g
