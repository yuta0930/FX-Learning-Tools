import numpy as np
from src.execution.executor import ExecutionEngine, ExecutionConfig
from src.risk.risk_manager import RiskManager, RiskConfig


def test_daily_loss_halts_new():
    rm = RiskManager(RiskConfig(daily_loss_limit_pct=0.02))
    rm.update_equity(-0.015)
    rm.update_equity(-0.01)
    assert rm.should_halt_new() is True


def test_cooldown_and_slippage():
    ee = ExecutionEngine(ExecutionConfig(cool_down_bars=2), rng=np.random.default_rng(0))
    assert ee.can_trade('NV') is True
    ee.on_fill()
    assert ee.can_trade('NV') is False
    ee.tick(); ee.tick()
    assert ee.can_trade('NV') is True
    p = 150.0
    p_b = ee.apply_slippage('buy', p, spread_proxy=0.01, realized_vol=0.02)
    p_s = ee.apply_slippage('sell', p, spread_proxy=0.01, realized_vol=0.02)
    assert p_b >= p and p_s <= p
