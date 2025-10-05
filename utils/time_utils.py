"""Time utilities: centralize timezone handling.

Exports:
- JST: Asia/Tokyo timezone
- to_jst(ts): convert pandas.Timestamp/Series/DatetimeIndex to JST
- ensure_tzaware(ts, tz): make a timestamp tz-aware (localize if naive, else convert)
"""
from __future__ import annotations

import pandas as pd
import pytz
from typing import Union

JST = pytz.timezone("Asia/Tokyo")


def ensure_tzaware(ts: pd.Timestamp, tz=pytz.UTC) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)


def to_jst(obj: Union[pd.Timestamp, pd.Series, pd.DatetimeIndex]) -> Union[pd.Timestamp, pd.Series, pd.DatetimeIndex]:
    if isinstance(obj, pd.Timestamp):
        return ensure_tzaware(obj, pytz.UTC).tz_convert(JST)
    if isinstance(obj, pd.Series) and isinstance(obj.dtype, pd.DatetimeTZDtype | type):
        try:
            return obj.dt.tz_convert(JST)
        except Exception:
            return obj.dt.tz_localize("UTC").dt.tz_convert(JST)
    if isinstance(obj, pd.DatetimeIndex):
        if obj.tz is None:
            return obj.tz_localize("UTC").tz_convert(JST)
        return obj.tz_convert(JST)
    # Fallback: try to convert if it's datetime-like
    try:
        s = pd.to_datetime(obj)
        if isinstance(s, pd.Series):
            return s.dt.tz_localize("UTC").dt.tz_convert(JST)
        if isinstance(s, pd.DatetimeIndex):
            return s.tz_localize("UTC").tz_convert(JST)
    except Exception:
        pass
    return obj
