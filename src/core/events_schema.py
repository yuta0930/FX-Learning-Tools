from __future__ import annotations

from typing import List, Optional, Literal
from datetime import date
from pydantic import BaseModel, field_validator
import re


TIME_RE = re.compile(r"^\d{2}:\d{2}$")  # HH:MM


class EventItemModel(BaseModel):
    title: str
    date: date
    time: Optional[str] = None  # "HH:MM" (JST既定)
    importance: Literal["low", "medium", "high"] = "medium"
    tz: str = "Asia/Tokyo"

    @field_validator("time")
    @classmethod
    def _time_format(cls, v: Optional[str]) -> Optional[str]:
        if v in (None, ""):
            return None
        assert TIME_RE.match(v), "time must be HH:MM (00-23:00-59)"
        return v


def validate_events_payload(payload: dict) -> List[EventItemModel]:
    items = payload.get("events", []) or []
    return [EventItemModel.model_validate(i) for i in items]
