import numpy as np
from dataclasses import dataclass

@dataclass
class RiskConfig:
    daily_loss_limit_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    kelly_cap: float = 0.25
    base_risk_per_trade_pct: float = 0.5
    reduce_on_dd: bool = True

class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.daily_pl = 0.0
        self.equity_peak = None
        self.equity = 1.0

    def reset_day(self):
        self.daily_pl = 0.0

    def update_equity(self, pnl_pct: float):
        self.equity *= (1.0 + pnl_pct)
        if self.equity_peak is None:
            self.equity_peak = self.equity
        else:
            self.equity_peak = max(self.equity_peak, self.equity)
        self.daily_pl += pnl_pct

    def drawdown_pct(self) -> float:
        if self.equity_peak is None or self.equity_peak <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.equity_peak)

    def _beyond_dd(self) -> bool:
        return self.drawdown_pct() >= self.cfg.max_drawdown_pct

    def should_halt_new(self) -> bool:
        return (self.daily_pl <= -self.cfg.daily_loss_limit_pct) or self._beyond_dd()

    def kelly_fraction(self, p: float, R: float) -> float:
        edge = p * R - (1.0 - p)
        if R <= 1e-9:
            return 0.0
        f = edge / R
        return float(np.clip(f, 0.0, self.cfg.kelly_cap))

    def position_size_pct(self, p: float, atr_distance: float, R: float) -> float:
        f = self.kelly_fraction(p, R)
        base = self.cfg.base_risk_per_trade_pct / 100.0
        denom = max(atr_distance, 1e-6)
        size = f * base / denom
        if self.cfg.reduce_on_dd and self._beyond_dd():
            size *= 0.5
        return max(0.0, float(size))
