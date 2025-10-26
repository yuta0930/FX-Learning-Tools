from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check


def market_schema() -> DataFrameSchema:
    """OHLC の簡易スキーマ。

    - time は任意（naive想定）
    - OHLC は数値に強制変換（coerce）して下限0を要求
    - 余分列は許容（strict=False）
    """

    return DataFrameSchema(
        {
            "time": Column(object, required=False, nullable=True),
            "open": Column(pa.Float, coerce=True, checks=[Check.ge(0)]),
            "high": Column(pa.Float, coerce=True, checks=[Check.ge(0)]),
            "low": Column(pa.Float, coerce=True, checks=[Check.ge(0)]),
            "close": Column(pa.Float, coerce=True, checks=[Check.ge(0)]),
        },
        coerce=True,
        strict=False,
    )


def validate_market_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """OHLC の存在/数値性を軽く検証。

    成功時: （型変換済みの）DataFrame と 空リスト
    失敗時: 元のDataFrame と 簡単なメッセージ群
    """

    schema = market_schema()
    try:
        v = schema.validate(df, lazy=True)
        # 追加のソフトチェック（警告のみ）
        msgs: List[str] = []
        if "time" in v.columns:
            try:
                t = pd.to_datetime(v["time"], errors="coerce")
                if t.isna().any():
                    msgs.append("time: could not parse some rows to datetime (coerce=NaT)")
                # tz-aware 混入チェック
                for x in t.dropna().tolist():
                    if getattr(x, "tzinfo", None) is not None:
                        msgs.append("time: timezone-aware values detected; expected naive JST")
                        break
            except Exception:
                pass
        return v, msgs
    except pa.errors.SchemaErrors as e:  # type: ignore[attr-defined]
        msgs: List[str] = []
        fc = getattr(e, "failure_cases", None)
        if fc is not None and isinstance(fc, pd.DataFrame):
            for _, row in fc.head(10).iterrows():
                col = str(row.get("column", ""))
                chk = str(row.get("check", ""))
                reason = str(row.get("failure_case", ""))
                msgs.append(f"column={col} check={chk} failure={reason}")
        else:
            msgs.append(str(e))
        return df, msgs
