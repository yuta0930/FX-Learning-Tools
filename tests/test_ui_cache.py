import io
import os
import time
from pathlib import Path

import pandas as pd

from src.ui.cache import file_mtime, read_csv_cached, read_parquet_cached


def test_csv_cache_invalidation(tmp_path: Path):
    p = tmp_path / "sample.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")

    m1 = file_mtime(p)
    df1 = read_csv_cached(str(p), m1)
    assert list(df1.columns) == ["a", "b"]
    assert int(df1.iloc[0, 0]) == 1

    # Update file and ensure mtime changes
    time.sleep(0.01)
    p.write_text("a,b\n3,4\n", encoding="utf-8")

    m2 = file_mtime(p)
    assert m2 != m1
    df2 = read_csv_cached(str(p), m2)

    assert int(df2.iloc[0, 0]) == 3


def test_parquet_cache_invalidation(tmp_path: Path):
    p = tmp_path / "sample.parquet"
    df = pd.DataFrame({"x": [1, 2, 3]})
    df.to_parquet(p)

    m1 = file_mtime(p)
    df1 = read_parquet_cached(str(p), m1)
    assert df1["x"].sum() == 6

    # Update parquet
    time.sleep(0.01)
    df2 = pd.DataFrame({"x": [4, 5]})
    df2.to_parquet(p)

    m2 = file_mtime(p)
    assert m2 != m1
    df3 = read_parquet_cached(str(p), m2)
    assert df3["x"].sum() == 9
