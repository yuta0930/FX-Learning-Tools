from __future__ import annotations


def minutes_from_interval(interval: str) -> int | None:
    """Convert interval string (e.g., '15m','60m','1d') to minutes if intraday."""
    try:
        s = str(interval).strip().lower()
        if s.endswith("m"):
            return int(s[:-1])
        if s.endswith("h"):
            return int(s[:-1]) * 60
        return None
    except Exception:
        return None


def bars_per_day(interval: str) -> int:
    """Bars per 24h day (used as fallback when timestamp is missing)."""
    m = minutes_from_interval(interval)
    if m and m > 0:
        return max(1, int(round(24 * 60 / m)))
    if str(interval).strip().lower() == "1d":
        return 1
    # Safe default: keep old behavior
    return 96


def recommended_refresh_secs(interval: str) -> int:
    """UI refresh default per timeframe (conservative)."""
    s = str(interval).strip().lower()
    if s in {"60m", "1h"}:
        return 600
    if s == "30m":
        return 300
    if s == "15m":
        return 180
    if s == "5m":
        return 60
    if s == "1d":
        return 600
    return 180
