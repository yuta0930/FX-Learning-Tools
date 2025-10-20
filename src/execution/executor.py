import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ExecutionConfig:
    slippage_alpha: float = 0.35
    slippage_beta: float = 0.25
    max_retry: int = 2
    cool_down_bars: int = 2
    allowed_regimes: List[str] = None

    def __post_init__(self):
        if self.allowed_regimes is None:
            self.allowed_regimes = ["HV_HLQ","HV","NV"]

class ExecutionEngine:
    def __init__(self, cfg: ExecutionConfig, rng: Optional[np.random.Generator]=None):
        self.cfg = cfg
        self.rng = np.random.default_rng() if rng is None else rng
        self.cooldown = 0

    def tick(self):
        if self.cooldown > 0:
            self.cooldown -= 1

    def can_trade(self, regime_name: str) -> bool:
        if self.cooldown > 0:
            return False
        return regime_name in self.cfg.allowed_regimes

    def apply_slippage(self, side: str, price: float, spread_proxy: float, realized_vol: float) -> float:
        loc = spread_proxy * self.cfg.slippage_alpha
        scale = max(1e-5, spread_proxy * self.cfg.slippage_beta + realized_vol * 0.05)
        slip = float(self.rng.normal(loc=loc, scale=scale))
        return price + abs(slip) if side.lower() in ("buy","long") else price - abs(slip)

    def on_fill(self):
        self.cooldown = max(self.cooldown, int(self.cfg.cool_down_bars))
