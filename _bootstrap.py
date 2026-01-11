"""Project-wide path bootstrap.

Some CLI scripts under `scripts/` do `import _bootstrap` so they can be executed
from various working directories while still importing the local `src` package.

Tests may import those scripts as modules; providing this file at the repository
root makes `import _bootstrap` resolve reliably.

This module is intentionally tiny and side-effect limited to sys.path edits.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _ensure_path(p: Path) -> None:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


_ensure_path(ROOT)
_ensure_path(ROOT / "src")
