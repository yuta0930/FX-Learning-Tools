import pandas as pd
from utils.app_core import build_event_windows_pure


def test_build_event_windows_pure_basic():
    df = pd.DataFrame({
        'time': pd.to_datetime(['2025-01-01 00:00Z', '2025-01-01 01:00Z']),
        'importance': [3, 5],
        'title': ['A', 'B'],
    })
    mapping = {3: 10, 5: 30}
    out = build_event_windows_pure(df, imp_threshold=3, mapping=mapping)
    assert list(out.columns) == ['start','end','importance','title']
    assert len(out) == 2
    assert (out.loc[0,'start'] == pd.Timestamp('2025-01-01 00:00Z') - pd.Timedelta(minutes=10))
    assert (out.loc[1,'end'] == pd.Timestamp('2025-01-01 01:00Z') + pd.Timedelta(minutes=30))


def test_build_event_windows_pure_filters_by_threshold():
    df = pd.DataFrame({
        'time': pd.to_datetime(['2025-01-01 00:00Z', '2025-01-01 01:00Z']),
        'importance': [2, 3],
    })
    mapping = {3: 10}
    out = build_event_windows_pure(df, imp_threshold=3, mapping=mapping)
    assert len(out) == 1
    assert out.iloc[0]['importance'] == 3


def test_build_event_windows_pure_handles_invalid_rows():
    df = pd.DataFrame({
        'time': [pd.NaT, pd.Timestamp('2025-01-01 01:00Z')],
        'importance': ['x', 5],
    })
    mapping = {5: 20}
    out = build_event_windows_pure(df, imp_threshold=3, mapping=mapping)
    assert len(out) == 1
    assert out.iloc[0]['importance'] == 5
