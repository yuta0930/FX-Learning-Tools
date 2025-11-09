from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import os
import hashlib
import yaml
import pandas as pd

from constants import TRADES_LOG, ARTIFACTS_DIR


@dataclass
class ABConfig:
    enabled: bool
    allocation: Dict[str, float]
    variants: Dict[str, Dict]
    eval: Dict


class ABManager:
    def __init__(self, cfg_path: str = "configs/ab.yml") -> None:
        self.cfg_path = cfg_path
        self.cfg = self._load()
        self._state_path = os.path.join(ARTIFACTS_DIR, "ab_last_promotion.txt")

    def _load(self) -> ABConfig:
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)["ab"]
        # Filter only known fields to avoid unexpected-key errors (e.g., promote_policy)
        allowed = {k: y.get(k) for k in ("enabled", "allocation", "variants", "eval")}
        return ABConfig(**allowed)

    def assign(self, setup_id: str) -> str:
        if not self.cfg.enabled:
            return "A"
        h = int(hashlib.sha256(setup_id.encode("utf-8")).hexdigest(), 16)
        r = (h % 10000) / 10000.0
        a = float(self.cfg.allocation.get("A", 0.5))
        return "A" if r < a else "B"

    def evaluate_and_promote(self) -> Tuple[str, Dict]:
        if not os.path.exists(TRADES_LOG):
            return "none", {}
        df = pd.read_parquet(TRADES_LOG)
        if "variant" not in df.columns:
            return "none", {}
        win = int(self.cfg.eval.get("window_trades", 300))
        df = df.tail(win)
        # Policy: minimum trades per arm
        pol = (yaml.safe_load(open(self.cfg_path, "r", encoding="utf-8")) or {}).get("ab", {}).get("promote_policy", {})
        min_arm = int(pol.get("min_trades_per_arm", 0) or 0)

        def metrics(g):
            n = len(g)
            gross_win = g.loc[g["pnl_pips"] > 0, "pnl_pips"].sum()
            gross_loss = -g.loc[g["pnl_pips"] <= 0, "pnl_pips"].sum()
            pf = (gross_win / max(gross_loss, 1e-6)) if n else 0.0
            ev = g["pnl_pips"].mean() if n else 0.0
            return pd.Series({"PF": pf, "EV": ev, "N": n})
        # Future-proof: avoid future pandas behavior change and warning.
        # group_keys=False keeps a flat index. include_groups=False (new pandas option) excludes grouping keys from data passed.
        try:
            m = df.groupby("variant", group_keys=False).apply(metrics, include_groups=False)
        except TypeError:  # include_groups not yet available in older pandas
            m = df.groupby("variant", group_keys=False).apply(metrics)
        if min_arm > 0:
            for arm in ("A", "B"):
                if arm not in m.index or m.loc[arm, "N"] < min_arm:
                    return "none", m.to_dict()
        if not {"A", "B"}.issubset(m.index):
            return "none", m.to_dict()
        primary = self.cfg.eval.get("metric_primary", "PF")
        eff = float(self.cfg.eval.get("min_effect_size", 0.2))
        promote_on_sig = bool(self.cfg.eval.get("promote_on_sig", True))
        a = m.loc["A", primary]
        b = m.loc["B", primary]
        winner = "A" if a >= b else "B"
        diff = abs(a - b)

        if promote_on_sig and diff >= eff:
            alloc = dict(self.cfg.allocation)
            step = float(pol.get("shift_step", 0.1))
            max_exploit = float(pol.get("max_exploit", 0.95))
            cooldown_h = float(pol.get("cooldown_hours", 0.0) or 0.0)
            import time as _time
            now = _time.time()
            try:
                if cooldown_h > 0 and os.path.exists(self._state_path):
                    last = float(open(self._state_path, "r", encoding="utf-8").read().strip() or 0.0)
                    if (now - last) < (cooldown_h * 3600.0):
                        return winner, m.to_dict()
            except Exception:
                pass
            if winner == "A":
                alloc["A"] = min(max_exploit, float(alloc.get("A", 0.5)) + step)
                alloc["B"] = 1.0 - alloc["A"]
            else:
                alloc["B"] = min(max_exploit, float(alloc.get("B", 0.5)) + step)
                alloc["A"] = 1.0 - alloc["B"]
            try:
                with open(self.cfg_path, "r", encoding="utf-8") as f:
                    y = yaml.safe_load(f)
                y["ab"]["allocation"] = alloc
                with open(self.cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False)
                self.cfg = self._load()
                os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
                with open(self._state_path, "w", encoding="utf-8") as f:
                    f.write(str(now))
            except Exception:
                pass

        return winner, m.to_dict()
