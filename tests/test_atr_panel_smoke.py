import pandas as pd
from datetime import datetime, timezone, timedelta


def _make_df(n: int = 10) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    times = [now - timedelta(minutes=15 * (n - i)) for i in range(n)]
    # simple synthetic series
    close = [150.0 + 0.1 * i for i in range(n)]
    atr = [0.12 for _ in range(n)]
    sess = ["London" for _ in range(n)]
    sp = [0.5 for _ in range(n)]  # pips
    return pd.DataFrame({
        "time": times,
        "close": close,
        "atr": atr,
        "session": sess,
        "spread_pips": sp,
    })


def test_render_atr_panel_smoke_no_exception():
    from src.ui.atr_panel import render_atr_panel  # import inside test

    df = _make_df(20)
    # Should not raise
    render_atr_panel(df=df)


def test_render_atr_panel_handles_no_data_path_gracefully(tmp_path):
    from src.ui.atr_panel import render_atr_panel

    # Non-existing path should be handled (no crash)
    render_atr_panel(data_path=str(tmp_path / "missing.csv"))
