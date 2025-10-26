from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any

from src.core.safety import trading_enabled


def _append_reason_dict(d: dict, reason: str) -> dict:
    """Append reason into dict as `reasons` (list) or `reason` (str) safely."""
    if "reasons" in d:
        rs = d["reasons"]
        if isinstance(rs, list):
            rs.append(reason)
        else:
            d["reasons"] = [rs, reason]
    elif "reason" in d:
        r = d["reason"]
        d["reasons"] = r + [reason] if isinstance(r, list) else [r, reason]
    else:
        d["reason"] = reason
    return d


def enforce_env_guard(decision: Any, reason: str = "disabled by MODE/KILL_SWITCH") -> Any:
    """
    Apply environment guard (MODE/KILL_SWITCH) to a decision-like object.

    - If decision is a dict: set d["allow"] = False when present and append reason(s)
    - If decision is a dataclass with field `allow`: return a new instance with allow=False
    - Otherwise: if object has attribute `allow`, set it to False (best-effort)
    - If trading is enabled by env, return decision unchanged
    """
    # If env allows trading, pass-through
    if trading_enabled():
        return decision

    # dict case
    if isinstance(decision, dict):
        d = dict(decision)
        if "allow" in d:
            d["allow"] = False
        d = _append_reason_dict(d, reason)
        return d

    # dataclass case
    if is_dataclass(decision):
        try:
            # type: ignore[arg-type]
            return replace(decision, allow=False)
        except Exception:
            return decision

    # generic object with `allow` attribute
    if hasattr(decision, "allow"):
        try:
            setattr(decision, "allow", False)
        except Exception:
            pass

    return decision
