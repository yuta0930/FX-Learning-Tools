import pandas as pd
from risk_guard import TradeGuard, RiskConfig

def test_daily_limit():
    cfg = RiskConfig(max_trades_per_day=2, max_trades_per_session=10, max_consecutive_losses=5)
    g = TradeGuard(cfg)
    ts = pd.Timestamp('2025-09-16 09:00:00', tz='UTC')
    assert g.allow_new_trade(ts, session='Tokyo')[0]
    g.register_trade(result_pips=+1, ts=ts, session='Tokyo')
    assert g.allow_new_trade(ts, session='Tokyo')[0]
    g.register_trade(result_pips=+1, ts=ts, session='Tokyo')
    # 3回目は拒否
    allowed, reason = g.allow_new_trade(ts, session='Tokyo')
    assert not allowed and reason == 'daily_trade_limit'


def test_session_limit():
    cfg = RiskConfig(max_trades_per_day=10, max_trades_per_session=1, max_consecutive_losses=5)
    g = TradeGuard(cfg)
    ts = pd.Timestamp('2025-09-16 10:00:00', tz='UTC')
    assert g.allow_new_trade(ts, session='Tokyo')[0]
    g.register_trade(result_pips=-1, ts=ts, session='Tokyo')
    allowed, reason = g.allow_new_trade(ts, session='Tokyo')
    assert not allowed and reason == 'session_trade_limit'


def test_consecutive_loss_cooldown():
    cfg = RiskConfig(max_trades_per_day=20, max_trades_per_session=20, max_consecutive_losses=3, cooldown_minutes=1)
    g = TradeGuard(cfg)
    base_ts = pd.Timestamp('2025-09-16 11:00:00', tz='UTC')
    for i in range(3):
        assert g.allow_new_trade(base_ts, session='Tokyo')[0]
        g.register_trade(result_pips=-1, ts=base_ts, session='Tokyo')
    allowed, reason = g.allow_new_trade(base_ts, session='Tokyo')
    assert not allowed and reason == 'in_cooldown'


def test_atr_spike():
    cfg = RiskConfig(max_trades_per_day=20, max_trades_per_session=20, max_consecutive_losses=5, atr_spike_window=5, atr_spike_zscore=1.0)
    g = TradeGuard(cfg)
    ts = pd.Timestamp('2025-09-16 12:00:00', tz='UTC')
    # 平常ATRを記録
    for v in [10,10,11,9,10]:
        g.record_atr(v)
    # スパイク値
    allowed, _ = g.allow_new_trade(ts, session='Tokyo', current_atr=20)
    assert not allowed


def test_state_contains_limits():
    cfg = RiskConfig()
    g = TradeGuard(cfg)
    st = g.state()
    assert 'max_day_trades' in st and 'max_session_trades' in st and 'loss_cooldown_after' in st
