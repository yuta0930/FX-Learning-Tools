import pandas as pd

from src.core.schema import validate_market_df


def test_market_schema_warns_on_missing_columns():
    df = pd.DataFrame({"open": [1, 2], "close": [1, 2]})  # high/low 欠落
    _, msgs = validate_market_df(df)
    assert msgs, "high/low 欠落で警告が発生するはず"


def test_market_schema_ok_on_minimal_valid_df():
    df = pd.DataFrame({"open": [1, 2], "high": [2, 3], "low": [0.5, 1.5], "close": [1.2, 2.1]})
    v, msgs = validate_market_df(df)
    assert not msgs
    assert list(v.columns)
