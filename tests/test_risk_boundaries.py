from app.risk.risk_manager import RiskManager


def test_daily_loss_cap_boundaries(tmp_path):
    cfg = {"gate": {"daily_loss_cap_pct": 0.03, "cooldown_on_losstreak": 3, "cooldown_minutes": 45}}
    art = tmp_path / "artifacts"
    rm = RiskManager(cfg, str(art))

    start = 1_000_000.0
    # 0.0299 → OK
    ok, _ = rm.allow_new_trade(start, start * (-0.0299), 0)
    assert ok
    # 0.0300 → NG
    ok, reason = rm.allow_new_trade(start, start * (-0.0300), 0)
    assert (not ok) and (reason == "risk_daily_cap")
    # 0.0301 → NG
    ok, reason = rm.allow_new_trade(start, start * (-0.0301), 0)
    assert (not ok)


def test_losstreak_cooldown(tmp_path):
    cfg = {"gate": {"daily_loss_cap_pct": 0.03, "cooldown_on_losstreak": 3, "cooldown_minutes": 1}}
    rm = RiskManager(cfg, str(tmp_path / "artifacts"))
    # 3連敗でクールダウンへ
    rm.on_trade_close(-1.0)
    rm.on_trade_close(-1.0)
    rm.on_trade_close(-1.0)
    ok, reason = rm.allow_new_trade(1_000_000.0, 0.0, 3)
    assert (not ok) and (reason == "losstreak_cooldown")
