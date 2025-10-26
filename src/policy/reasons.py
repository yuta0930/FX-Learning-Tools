from __future__ import annotations

from typing import Tuple
import pandas as pd


def unify_reasons_df(
    df: pd.DataFrame,
    *,
    src_cols: Tuple[str, ...] = ("gate_reason", "deny_reason"),
    out_col: str = "reason",
    sep: str = " | ",
) -> pd.DataFrame:
    """複数の理由列を1列に統合。存在する列だけを対象にし、無ければNo-Op。

    - 文字列以外は文字列化。
    - 空文字やNaNは無視して結合。
    - out_col を追加（既存は温存）。
    """
    present = [c for c in src_cols if c in df.columns]
    if not present:
        return df

    out = df.copy()
    # すべて文字列化し、NaN->""
    vals = out[present].astype("string").fillna("")
    # 行単位で非空のみ sep 結合
    merged = vals.apply(lambda row: sep.join([x for x in row if x]), axis=1)
    out[out_col] = merged.str.strip()
    return out
