from utils.timeframe_profile import bars_per_day, recommended_refresh_secs


def test_bars_per_day_intraday():
    assert bars_per_day("15m") == 96
    assert bars_per_day("60m") == 24
    assert bars_per_day("30m") == 48
    assert bars_per_day("5m") == 288


def test_bars_per_day_daily():
    assert bars_per_day("1d") == 1


def test_recommended_refresh_secs():
    assert recommended_refresh_secs("15m") == 180
    assert recommended_refresh_secs("60m") == 600

