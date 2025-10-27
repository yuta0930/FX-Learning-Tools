from __future__ import annotations
from datetime import datetime, timezone
from typing import Iterable
import pytz

TZ_NAME = "Asia/Tokyo"
JST = pytz.timezone(TZ_NAME)


def now_jst() -> datetime:
    """Return current datetime in JST timezone (aware)."""
    return datetime.now(tz=JST)


def to_jst(dt: datetime) -> datetime:
    """Convert an aware/naive datetime to JST (aware). Naive assumed UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(JST)


def iso_jst(dt: datetime | None = None) -> str:
    d = to_jst(dt or now_jst())
    return d.isoformat()


def session_from_ts(ts: datetime) -> str:
    """Naive session classifier for JST timestamps: tokyo/london/ny"""
    h = to_jst(ts).hour
    if 9 <= h < 15:
        return "tokyo"
    if 16 <= h < 24:
        return "london"
    return "ny"
