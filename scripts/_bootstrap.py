"""Lightweight path bootstrap so scripts can import the local `src` package.

This keeps CLI scripts runnable both from the repository root and when invoked
from elsewhere (e.g., scheduled tasks, CI). The parent of this file is the
project root containing `src/`.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

def _ensure_path(p: Path) -> None:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


_ensure_path(ROOT)
_ensure_path(ROOT / "src")
