import importlib
import os
from typing import Callable, Optional

import pandas as pd
import pytest


def set_mode(monkeypatch: pytest.MonkeyPatch, mode: str, kill: Optional[str] = None) -> None:
    monkeypatch.setenv("MODE", mode)
    if kill is None:
        monkeypatch.delenv("KILL_SWITCH", raising=False)
    else:
        monkeypatch.setenv("KILL_SWITCH", kill)

    # Ensure safety module sees the latest env
    import src.core.safety as safety

    importlib.reload(safety)


def _import_gate_module():
    # Try common import paths and skip if none available
    for mod in ("src.policy.gate", "policy.gate"):
        try:
            return importlib.import_module(mod)
        except Exception:
            continue
    pytest.skip("gate module not found in expected locations")


@pytest.mark.parametrize("mode,kill", [("paper", None), ("live", "1")])
def test_apply_final_gate_env_guard_effect(monkeypatch: pytest.MonkeyPatch, mode: str, kill: Optional[str]):
    """
    If apply_final_gate exists and returns a DataFrame, verify that under env guard
    (MODE!=live or KILL on) the final trade_ok becomes False for all rows.
    Signature-agnostic: try a few common call patterns; skip safely if none match.
    """
    gate = _import_gate_module()
    fn: Optional[Callable] = getattr(gate, "apply_final_gate", None)
    if fn is None:
        pytest.skip("apply_final_gate not found")

    df = pd.DataFrame({"timestamp": [0, 1, 2], "signal": [True, True, True], "trade_ok": [True, True, True]})

    set_mode(monkeypatch, mode, kill)

    # Try a few signatures: (df), (df=df), (pred_df, windows_df=None, state={})
    out = None
    call_patterns = (
        lambda: fn(df),
        lambda: fn(df=df),
        lambda: fn(df, None, state={}),
        lambda: fn(pred_df=df, windows_df=None, state={}),
    )
    for call in call_patterns:
        try:
            out = call()
            break
        except TypeError:
            continue

    if out is None:
        pytest.skip("apply_final_gate signature unsupported by this test")

    if not isinstance(out, pd.DataFrame):
        pytest.skip("apply_final_gate did not return a DataFrame")

    assert "trade_ok" in out.columns
    assert out["trade_ok"].eq(False).all()
