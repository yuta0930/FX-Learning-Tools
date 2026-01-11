import pandas as pd
from utils.time_utils import JST, to_jst, ensure_tzaware


def test_ensure_tzaware_localizes_naive_to_utc():
    ts = pd.Timestamp('2025-01-01 00:00:00')  # naive
    out = ensure_tzaware(ts)
    assert out.tz is not None
    assert out.tz.zone in ('UTC', 'UTC')


def test_to_jst_timestamp_and_series():
    ts_utc = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
    ts_jst = to_jst(ts_utc)
    assert ts_jst.tz == JST
    # 9時間進む
    assert ts_jst.hour == (ts_utc.hour + 9) % 24

    s = pd.Series([ts_utc, ts_utc + pd.Timedelta(minutes=15)])
    s_jst = to_jst(s)
    assert hasattr(s_jst, 'dt')
    assert str(s_jst.dt.tz) == str(JST)
    assert (s_jst.dt.hour.iloc[0] == (ts_utc.hour + 9) % 24)


def test_to_jst_datetime_index():
    idx = pd.date_range('2025-01-01', periods=2, freq='1h', tz='UTC')
    out = to_jst(idx)
    assert out.tz == JST
    assert (out.hour == ((idx.hour + 9) % 24)).all()
