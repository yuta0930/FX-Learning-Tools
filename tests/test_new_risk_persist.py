import os
import json
from app.risk.risk_manager import RiskManager, RiskConfig
import constants as C


def test_persisted_risk_daily_cap_and_cooldown(tmp_path, monkeypatch):
    # Redirect ARTIFACTS_DIR to temp
    new_art = tmp_path / "artifacts"
    os.makedirs(new_art / "risk", exist_ok=True)
    monkeypatch.setattr(C, "ARTIFACTS_DIR", str(new_art), raising=False)

    rm = RiskManager(RiskConfig(daily_loss_cap_pct=0.01, cooldown_on_losstreak=2, cooldown_minutes=1))
    # initialize day start equity and no pnl
    ok, reason = rm.allow_new_trade(equity_day_start_ccy=1000000.0, realized_pnl_today_ccy=0.0, current_losstreak=0)
    assert ok and reason == "ok"
    # two losing trades cause cooldown or daily cap
    rm.on_trade_close(-1000.0)
    rm.on_trade_close(-2000.0)
    ok2, reason2 = rm.allow_new_trade(equity_day_start_ccy=None, realized_pnl_today_ccy=None, current_losstreak=None)
    assert (not ok2) and reason2 in {"losstreak_cooldown", "risk_daily_cap"}
    # also ensure state file exists
    files = os.listdir(os.path.join(str(new_art), "risk"))
    assert any(f.startswith("state_") and f.endswith(".json") for f in files)
