"""Risk / trade guard utilities.

Provides:
  - Per-day & per-session trade count limits
  - Consecutive loss cooldown
  - ATR spike (volatility shock) new-trade halt

Integration points (typical):
  1. Initialize a TradeGuard once (streamlit session or process global)
  2. Before placing a trade, call guard.allow_new_trade(timestamp, session_name)
  3. After a trade is closed / outcome known, call guard.register_trade(result_pips)
  4. Optionally display guard.state() in the UI

State is in-memory; for persistence across restarts you could serialize guard.snapshot().
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional
import pandas as pd
import numpy as np
import time

@dataclass
class RiskConfig:
    max_trades_per_day: int = 40
    max_trades_per_session: int = 20
    max_consecutive_losses: int = 6
    cooldown_minutes: int = 60
    atr_spike_window: int = 20
    atr_spike_zscore: float = 3.0
    daily_reset_hour_utc: int = 0
    enable_atr_guard: bool = True
    enable_session_limits: bool = True
    enable_cooldown: bool = True
    atr_period: int = 14


class TradeGuard:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.day_key: Optional[str] = None
        self.day_trades: int = 0
        self.session_trades: Dict[str, int] = {}
        self.consecutive_losses: int = 0
        self.cooldown_until: float = 0.0  # epoch seconds
        self.last_prices: List[float] = []  # for ATR spike detection external feed
        self.atr_series: List[float] = []

    # ---------- public API ----------
    def allow_new_trade(self, ts: pd.Timestamp, session: Optional[str] = None, current_atr: Optional[float] = None) -> tuple[bool, str]:
        """発注前チェック。

        Args:
            ts: 現在時刻 (timezone aware 推奨)
            session: セッション名 (例: 'Tokyo','London','NY')。None許容。
            current_atr: ATR値（ATRガード評価に利用、Noneならスキップ）
        Returns:
            (allow: bool, reason: str)
        reason 値:
            ok / in_cooldown / daily_trade_limit / session_trade_limit / atr_spike_halt
        """
        self._maybe_reset_day(ts)
        # Cooldown check
        now_epoch = time.time()
        if self.cfg.enable_cooldown and now_epoch < self.cooldown_until:
            return False, "in_cooldown"
        # Daily limit
        if self.day_trades >= self.cfg.max_trades_per_day:
            return False, "daily_trade_limit"
        # Session limit
        if self.cfg.enable_session_limits and session is not None:
            if self.session_trades.get(session, 0) >= self.cfg.max_trades_per_session:
                return False, "session_trade_limit"
        # ATR spike guard
        if self.cfg.enable_atr_guard and current_atr is not None:
            if self._atr_spike(current_atr):
                return False, "atr_spike_halt"
        return True, "ok"

    def register_trade(self, result_pips: Optional[float], ts: Optional[pd.Timestamp] = None, session: Optional[str] = None):
        if ts is not None:
            self._maybe_reset_day(ts)
        self.day_trades += 1
        if session:
            self.session_trades[session] = self.session_trades.get(session, 0) + 1
        if result_pips is not None:
            if result_pips <= 0:
                self.consecutive_losses += 1
                if self.cfg.enable_cooldown and self.consecutive_losses >= self.cfg.max_consecutive_losses:
                    self.cooldown_until = time.time() + self.cfg.cooldown_minutes * 60
                    self.consecutive_losses = 0  # reset after triggering
            else:
                self.consecutive_losses = 0

    def record_atr(self, atr_value: float, max_store: int = 500):
        if not np.isfinite(atr_value):
            return
        self.atr_series.append(float(atr_value))
        if len(self.atr_series) > max_store:
            self.atr_series = self.atr_series[-max_store:]

    def state(self) -> Dict:
        return {
            "day_trades": self.day_trades,
            "session_trades": dict(self.session_trades),
            "consecutive_losses": self.consecutive_losses,
            "in_cooldown": time.time() < self.cooldown_until,
            "cooldown_remaining_min": max(0, int((self.cooldown_until - time.time())/60)) if time.time() < self.cooldown_until else 0,
            "atr_spike_last": self._last_spike_value(),
            # config limits for UI transparency
            "max_day_trades": self.cfg.max_trades_per_day,
            "max_session_trades": self.cfg.max_trades_per_session,
            "loss_cooldown_after": self.cfg.max_consecutive_losses,
            "atr_period": self.cfg.atr_period,
        }

    def snapshot(self) -> Dict:
        return {
            "cfg": asdict(self.cfg),
            "runtime": self.state(),
        }

    # ---------- internal helpers ----------
    def _maybe_reset_day(self, ts: pd.Timestamp):
        key = ts.tz_convert("UTC").strftime("%Y-%m-%d") if ts.tzinfo else ts.tz_localize("UTC").strftime("%Y-%m-%d")
        if self.day_key is None:
            self.day_key = key
        if key != self.day_key:
            # day rollover
            self.day_key = key
            self.day_trades = 0
            self.session_trades.clear()
            self.consecutive_losses = 0
            self.cooldown_until = 0.0

    def _atr_spike(self, current_atr: float) -> bool:
        w = self.cfg.atr_spike_window
        if len(self.atr_series) < w:
            return False
        window = np.array(self.atr_series[-w:], float)
        m = window.mean(); s = window.std(ddof=0)
        if s < 1e-9:
            return False
        z = (current_atr - m) / s
        return z >= self.cfg.atr_spike_zscore

    def _last_spike_value(self) -> float:
        if len(self.atr_series) == 0:
            return float('nan')
        return float(self.atr_series[-1])

    def update_config(self, cfg: RiskConfig):
        """Replace guard limits at runtime (state counters stay intact)."""
        self.cfg = cfg


def make_guard_from_config(cfg_node) -> TradeGuard:
    rc = RiskConfig(
        max_trades_per_day=cfg_node.max_trades_per_day,
        max_trades_per_session=cfg_node.max_trades_per_session,
        max_consecutive_losses=cfg_node.max_consecutive_losses,
        cooldown_minutes=cfg_node.cooldown_minutes,
        atr_spike_window=cfg_node.atr_spike_window,
        atr_spike_zscore=cfg_node.atr_spike_zscore,
        daily_reset_hour_utc=cfg_node.daily_reset_hour_utc,
        enable_atr_guard=cfg_node.enable_atr_guard,
        enable_session_limits=cfg_node.enable_session_limits,
        enable_cooldown=cfg_node.enable_cooldown,
        atr_period=getattr(cfg_node, 'atr_period', 14),
    )
    return TradeGuard(rc)


def clone_risk_config(cfg: RiskConfig) -> RiskConfig:
    """Create a deep copy of a RiskConfig dataclass."""
    return RiskConfig(**asdict(cfg))


def merge_risk_config(base: RiskConfig, overrides: Mapping[str, Any] | None) -> RiskConfig:
    """Return a new RiskConfig by applying overrides onto a base config."""
    if not overrides:
        return clone_risk_config(base)
    merged = asdict(base)
    for key, value in overrides.items():
        if key not in merged or value is None:
            continue
        current = merged[key]
        try:
            if isinstance(current, bool):
                merged[key] = bool(value)
            elif isinstance(current, int) and not isinstance(current, bool):
                merged[key] = int(value)
            elif isinstance(current, float):
                merged[key] = float(value)
            else:
                merged[key] = value
        except Exception:
            # Ignore invalid overrides to preserve safety
            continue
    return RiskConfig(**merged)


def apply_risk_guard_overrides(
    guard: TradeGuard,
    *,
    base_cfg: RiskConfig,
    overrides: Mapping[str, Any] | None,
) -> RiskConfig:
    """Apply overrides to a guard using the provided base config and return the new config."""
    new_cfg = merge_risk_config(base_cfg, overrides)
    guard.update_config(new_cfg)
    return new_cfg
