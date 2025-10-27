from __future__ import annotations
from typing import Tuple, Dict, Any
import os
import yaml
import numpy as np

from constants import CALIBRATION_DIR, SIGNALS_LOG, LOG_DIR
from app.calibration.online_calibrator import OnlineCalibrator
from app.ab.ab_manager import ABManager
from app.drift.drift_monitor import DriftMonitor
from app.tca.slippage_model import SlippageModel
from app.utils.io import append_parquet
from app.utils.timeutil import now_jst
from app.risk.risk_manager import RiskManager, RiskConfig


# --- Calibration ---
def calibrate_proba(p_raw: float | np.ndarray) -> float | np.ndarray:
    cal = OnlineCalibrator.load_current()
    if cal is None:
        return p_raw
    return cal.predict(p_raw)  # type: ignore


# --- AB assignment ---
def assign_ab_variant(setup_id: str) -> str:
    mgr = ABManager()
    return mgr.assign(setup_id)


# --- Gate decision ---
def gate_decision(signal_row: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Apply gate using configs/gate.yml and optional drift adjustment.
    signal_row keys expected: p_cal, quality, session, atr_15m, spread, news_flag, variant
    """
    cfg = yaml.safe_load(open("configs/gate.yml", "r", encoding="utf-8"))
    # support both styles: {gate:{...}} and flat keys
    gate = cfg.get("gate", cfg if isinstance(cfg, dict) else {})

    # Optional: apply drift adjustment from cached status file
    try:
        drift_link = gate.get("drift_link", {}) if isinstance(gate.get("drift_link", {}), dict) else {}
        if drift_link.get("apply_adjustment", False):
            import os as _os
            _p = _os.path.join(LOG_DIR, "drift_status.yml")
            if _os.path.exists(_p):
                _ds = yaml.safe_load(open(_p, "r", encoding="utf-8")) or {}
                _level = str(_ds.get("level", "ok")).lower()
                _adjust = _ds.get("adjust", {}) or {}
                if _level == "halt":
                    return False, {"checks": ["drift_halt"], "reason": "drift_halt"}
                if _level == "warn":
                    # tighten thresholds temporarily
                    gate["min_quality"] = float(gate.get("min_quality", 0.75)) + float(_adjust.get("quality_add", 0.0))
                    gate["p_threshold_break"] = float(gate.get("p_threshold_break", 0.56)) + float(_adjust.get("p_add", 0.0))
    except Exception:
        pass
    if not gate.get("enabled", True):
        return True, {"reason": "gate_disabled"}
    # Safety: missing artifacts handling (optional)
    safety = gate.get("safety", {}) if isinstance(gate.get("safety", {}), dict) else {}
    on_missing = str(safety.get("on_missing_artifacts", "continue")).lower()
    cur_cal = os.path.join(CALIBRATION_DIR, "current.joblib")
    if on_missing in {"halt", "stop"} and not os.path.exists(cur_cal):
        return False, {"checks": ["missing_calibrator"], "reason": "safety_on_missing_artifacts"}
    session = str(signal_row.get("session", "tokyo")).lower()
    min_q = float(gate.get("min_quality_tokyo", gate.get("min_quality", 0.75))) if session == "tokyo" else float(gate.get("min_quality", 0.75))
    p_th = float(gate.get("p_threshold_break", 0.56))
    # drift adjust (soft): optional baseline/current arrays not provided here; assume external monitor populates
    # basic checks
    meta: Dict[str, Any] = {"checks": []}
    if float(signal_row.get("quality", 0.0)) < min_q:
        meta["checks"].append("quality")
        return False, meta
    if float(signal_row.get("p_cal", 0.0)) < p_th:
        meta["checks"].append("p_cal")
        return False, meta
    if float(signal_row.get("spread", 0.0)) > float(gate.get("spread_max", 0.30)):
        meta["checks"].append("spread")
        return False, meta
    if bool(signal_row.get("news_flag", False)):
        meta["checks"].append("news")
        return False, meta
    # ATR based ksl/ktp by variant (returned as meta to be used by caller)
    ab = yaml.safe_load(open("configs/ab.yml", "r", encoding="utf-8"))
    var = str(signal_row.get("variant", "A"))
    vconf = ((ab or {}).get("ab", {}).get("variants", {}) or {}).get(var, {})
    meta.update({"ksl_atr": vconf.get("ksl_atr"), "ktp_atr": vconf.get("ktp_atr")})
    # Slippage guard flags for caller (execution stage)
    meta.update({
        "slippage_guard_ok": bool(gate.get("use_slippage_guard", True)),
        "ev_pred_min": float(gate.get("ev_pred_min", 0.0)),
    })
    return True, meta


# --- Slippage prediction ---
def predict_slippage(feats_row: Dict[str, Any]) -> float:
    """Predict slippage using configured percentile (p50 default, p90 if configured).

    Reads configs/gate.yml gate.slippage_guard.pctl to choose 50 or 90.
    """
    m = SlippageModel.load()
    if m is None:
        return 0.0
    pred = m.predict(feats_row)
    try:
        cfg = yaml.safe_load(open("configs/gate.yml", "r", encoding="utf-8"))
        gate = cfg.get("gate", cfg if isinstance(cfg, dict) else {})
        sg = gate.get("slippage_guard", {}) if isinstance(gate.get("slippage_guard", {}), dict) else {}
        pctl = int(sg.get("pctl", 50))
    except Exception:
        pctl = 50
    if pctl >= 90:
        return float(pred.get("p90", pred.get("p50", 0.0)))
    return float(pred.get("p50", 0.0))


# --- Risk checks (daily loss cap / loss-streak cooldown with persistence) ---
_RISK_MGR: RiskManager | None = None


def _get_risk_mgr() -> RiskManager:
    global _RISK_MGR
    if _RISK_MGR is None:
        try:
            g = yaml.safe_load(open("configs/gate.yml", "r", encoding="utf-8")) or {}
            gate = g.get("gate", g if isinstance(g, dict) else {})
            rc = RiskConfig(
                daily_loss_cap_pct=float(gate.get("daily_loss_cap_pct", 0.03)),
                cooldown_on_losstreak=int(gate.get("cooldown_on_losstreak", 3)),
                cooldown_minutes=int(gate.get("cooldown_minutes", 45)),
            )
        except Exception:
            rc = RiskConfig()
        _RISK_MGR = RiskManager(rc)
    return _RISK_MGR


def risk_check(event: str, *, equity_day_start_ccy: float | None = None,
               realized_pnl_today_ccy: float | None = None,
               current_losstreak: int | None = None) -> tuple[bool, str]:
    """Pre-entry risk check. Returns (ok, reason).

    event: "pre_entry" expected (others are no-op pass-through)
    The optional values allow callers to pass current equity baseline and realized PnL.
    """
    if event != "pre_entry":
        return True, "noop"
    rm = _get_risk_mgr()
    ok, reason = rm.allow_new_trade(equity_day_start_ccy, realized_pnl_today_ccy, current_losstreak)
    if not ok:
        # also log decision into signals log for auditability
        try:
            append_parquet({
                "ts_jst": now_jst(),
                "gate": "blocked",
                "gate_meta": {"risk_reason": reason},
                "run_id": None,
            }, SIGNALS_LOG)
        except Exception:
            pass
    return ok, reason


def risk_on_close(pnl_ccy: float | None) -> None:
    try:
        rm = _get_risk_mgr()
        rm.on_trade_close(pnl_ccy)
    except Exception:
        pass
