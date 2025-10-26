from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, List, Optional
from pathlib import Path

import pandas as pd
import yaml
from .events_schema import validate_events_payload, EventItemModel


JST = ZoneInfo("Asia/Tokyo")


@dataclass
class EventItem:
    title: str
    date: str  # "YYYY-MM-DD"
    time: Optional[str] = None  # "HH:MM" (JST 既定)
    importance: str = "medium"
    tz: str = "Asia/Tokyo"


def _parse_item(d: dict[str, Any]) -> EventItem:
    return EventItem(
        title=str(d["title"]),
        date=str(d["date"]),
        time=(str(d.get("time")) if d.get("time") not in (None, "") else None),
        importance=str(d.get("importance", "medium")),
        tz=str(d.get("tz", "Asia/Tokyo")),
    )


def load_events_yaml(path: Path) -> List[EventItem]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    valid: List[EventItemModel] = []
    try:
        valid = validate_events_payload(data)
    except Exception:
        # バリデーションに失敗しても既存互換: 可能な限り従来処理で読み込む（No-Op）
        items = data.get("events", [])
        return [_parse_item(x) for x in items]

    # 正常系: 型を明示して返す
    out: List[EventItem] = []
    for it in valid:
        out.append(
            EventItem(
                title=it.title,
                date=str(it.date),
                time=it.time,
                importance=it.importance,
                tz=it.tz,
            )
        )
    return out


def _to_jst_naive(date_str: str, time_str: Optional[str], tz: str) -> datetime:
    base_tz = ZoneInfo(tz) if tz else JST
    hhmm = time_str if time_str else "00:00"
    # パース → 元TZ付与 → JSTに変換 → tz情報除去（naive）
    dt = datetime.strptime(f"{date_str} {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=base_tz)
    dt_jst = dt.astimezone(JST)
    return dt_jst.replace(tzinfo=None)


def events_to_df(items: List[EventItem]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for it in items:
        dt = _to_jst_naive(it.date, it.time, it.tz)
        rows.append(
            {
                "time": dt.strftime("%Y-%m-%d %H:%M"),
                "title": it.title,
                "importance": it.importance,
            }
        )
    return pd.DataFrame(rows, columns=["time", "title", "importance"]) if rows else pd.DataFrame(
        columns=["time", "title", "importance"]
    )


def write_events_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
