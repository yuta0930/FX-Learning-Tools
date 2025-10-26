from datetime import datetime
import pandas as pd

from src.core.session import add_session


def test_add_session_basic():
    df = pd.DataFrame({
        "time": [datetime(2025, 1, 1, 10), datetime(2025, 1, 1, 17), datetime(2025, 1, 1, 2)]
    })
    out = add_session(df)
    assert set(out["session"]) == {"Tokyo", "London", "NewYork"}
