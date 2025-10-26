from __future__ import annotations

from typing import Optional
import pandas as pd


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trade_ok" not in out.columns:
        out["trade_ok"] = True
    if "deny_reason" not in out.columns:
        out["deny_reason"] = ""
    return out


def _append_reason(s: pd.Series, reason: str) -> pd.Series:
    s = s.astype("object")
    # 既存の文字列/NaNに追記。配列は想定しない（単純化）
    return s.where(s.isna() | (s == ""), other=s + " | " + reason).fillna(reason).replace("", reason)


def apply_spread_filter(
    df: pd.DataFrame, *, max_spread_pips: float, spread_col: str = "spread_pips"
) -> pd.DataFrame:
    """
    spread_col（例: spread_pips）が max_spread_pips を超える行を不可に。
    必要列が無ければ無変更（安全No-Op）。
    """
    if spread_col not in df.columns:
        return df

    out = _ensure_cols(df)
    mask_bad = out[spread_col] > max_spread_pips
    if mask_bad.any():
        out.loc[mask_bad, "trade_ok"] = False
        out.loc[mask_bad, "deny_reason"] = _append_reason(
            out.loc[mask_bad, "deny_reason"], f"spread>{max_spread_pips}"
        )
    return out


def apply_atr_filter(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr",
    min_atr: Optional[float] = None,
    max_atr: Optional[float] = None,
) -> pd.DataFrame:
    """
    ATRが範囲外（min未満 or max超過）を不可に。必要列が無ければ無変更。
    """
    if atr_col not in df.columns:
        return df

    out = _ensure_cols(df)
    if min_atr is not None:
        mask = out[atr_col] < float(min_atr)
        if mask.any():
            out.loc[mask, "trade_ok"] = False
            out.loc[mask, "deny_reason"] = _append_reason(
                out.loc[mask, "deny_reason"], f"atr<{min_atr}"
            )

    if max_atr is not None:
        mask = out[atr_col] > float(max_atr)
        if mask.any():
            out.loc[mask, "trade_ok"] = False
            out.loc[mask, "deny_reason"] = _append_reason(
                out.loc[mask, "deny_reason"], f"atr>{max_atr}"
            )

    return out


def apply_news_window_filter(
    df: pd.DataFrame,
    *,
    events: pd.DataFrame,
    minutes_before: int = 30,
    minutes_after: int = 30,
    time_col: str = "time",
    event_time_col: str = "time",
) -> pd.DataFrame:
    """
    指定イベントの前後ウィンドウで不可に。
    df[time_col], events[event_time_col] は日時。列が無ければ無変更。
    """
    if time_col not in df.columns or event_time_col not in events.columns or events.empty:
        return df

    out = _ensure_cols(df)
    t = pd.to_datetime(out[time_col], utc=False, errors="coerce")
    ev = pd.to_datetime(events[event_time_col], utc=False, errors="coerce").dropna().unique()
    if len(ev) == 0:
        return df

    # 各イベントのウィンドウに入るか判定（ベクトル化の簡易版）
    bad = pd.Series(False, index=out.index)
    for et in ev:
        bad = bad | ((t >= et - pd.Timedelta(minutes=minutes_before)) & (t <= et + pd.Timedelta(minutes=minutes_after)))

    if bad.any():
        out.loc[bad, "trade_ok"] = False
        out.loc[bad, "deny_reason"] = _append_reason(
            out.loc[bad, "deny_reason"], f"news_window±{minutes_before}/{minutes_after}m"
        )

    return out