from __future__ import annotations
import time
import uuid
from typing import Optional


def new_run_id(prefix: str = "run") -> str:
    # time-based unique id
    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"


def new_order_id(prefix: str = "ord") -> str:
    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"


# alias to match sample code in integration snippets
def run_id() -> str:
    return new_run_id()
