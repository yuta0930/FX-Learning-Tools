from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_failure(logs_dir: str, reason: str, details: Dict[str, Any] | None = None, hint: str | None = None) -> None:
    """Append a lightweight failure record to logs/gate_failures.jsonl.

    This is a non-intrusive helper that can be called from gates or checks
    to capture why something was blocked and what to do next.
    """
    os.makedirs(logs_dir, exist_ok=True)
    rec = {
        "ts": _ts(),
        "reason": reason,
        "details": details or {},
        "hint": hint,
    }
    path = os.path.join(logs_dir, "gate_failures.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
