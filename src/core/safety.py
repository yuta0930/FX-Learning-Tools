from __future__ import annotations

import os
from pathlib import Path

# Directory and file used to hard stop trading when needed
FLAGS_DIR = Path("flags")
KILL_FILE = FLAGS_DIR / "kill.switch"


def current_mode() -> str:
    """Return current mode ('paper' or 'live'). Defaults to 'paper' for safety."""
    return os.getenv("MODE", "paper").lower()


def _truthy(val: str | None) -> bool:
    if val is None:
        return False
    return str(val).lower() not in ("0", "", "false", "off", "no")


def is_kill_switch_on() -> bool:
    """Triple guard: env var and kill-switch file. If any guard is active, trading must be disabled."""
    env_on = _truthy(os.getenv("KILL_SWITCH", "0"))
    file_on = KILL_FILE.exists()
    return bool(env_on or file_on)


def trading_enabled() -> bool:
    """Trading is enabled only when MODE == 'live' and kill switch is OFF."""
    return current_mode() == "live" and not is_kill_switch_on()


def ensure_flags_dir() -> None:
    FLAGS_DIR.mkdir(parents=True, exist_ok=True)
