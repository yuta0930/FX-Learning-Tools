import pandas as pd
from policy.gate import apply_final_gate


def test_gate_allows_when_enabled_and_no_blocks():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=3, freq='15min', tz='UTC'),
        'signal': [True, True, True],
    })
    state = {
        'enable_trading': True,
        'auto_pause_on_drift': True,
        'drift_state': 'normal',
        'apply_news_filter': False,
        'guard_state': {'in_cooldown': False},
    }
    out = apply_final_gate(df, None, state=state)
    assert out['trade_ok'].all()
    assert (out['gate_reason'] == 'ok').all()


def test_gate_blocks_on_drift_alert():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=2, freq='15min', tz='UTC'),
        'signal': [True, True],
    })
    state = {
        'enable_trading': True,
        'auto_pause_on_drift': True,
        'drift_state': 'alert',
        'apply_news_filter': False,
        'guard_state': {'in_cooldown': False},
    }
    out = apply_final_gate(df, None, state=state)
    assert (~out['trade_ok']).all()
    assert (out['gate_reason'] == 'drift_block').all()


def test_gate_blocks_in_news_window():
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2025-01-01 00:00Z','2025-01-01 00:10Z']),
        'signal': [True, True],
    })
    windows = pd.DataFrame({
        'start': pd.to_datetime(['2025-01-01 00:00Z']),
        'end': pd.to_datetime(['2025-01-01 00:05Z'])
    })
    state = {
        'enable_trading': True,
        'auto_pause_on_drift': True,
        'drift_state': 'normal',
        'apply_news_filter': True,
        'guard_state': {'in_cooldown': False},
    }
    out = apply_final_gate(df, windows, state=state)
    assert out.loc[0,'trade_ok'] == False and out.loc[1,'trade_ok'] == True
    assert out.loc[0,'gate_reason'] == 'news_block'
