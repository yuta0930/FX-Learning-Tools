from __future__ import annotations
from typing import Dict, Any, Tuple
import yaml
from app.utils.ids import new_run_id
from app.utils.timeutil import now_jst
from app.utils.io import append_parquet
from constants import SIGNALS_LOG, ORDERS_LOG, TRADES_LOG
from components import calibrate_proba, assign_ab_variant, gate_decision, predict_slippage, risk_check, risk_on_close


def handle_signal(signal: Dict[str, Any], *, run_id: str | None = None) -> Tuple[bool, Dict[str, Any]]:
    """Apply post-signal hooks: calibration, A/B, gate. Append to signals log.

    Returns (ok, gate_meta). Mutates a shallow copy of signal to include p_cal/variant.
    """
    s = dict(signal)
    s["p_cal"] = calibrate_proba(s.get("p_raw"))
    s["variant"] = assign_ab_variant(str(s.get("setup_id", "")))
    ok, gate_meta = gate_decision({
        "p_cal": s.get("p_cal"),
        "variant": s.get("variant"),
        "atr_15m": s.get("atr_15m"),
        "session": s.get("session"),
        "spread": s.get("spread", 0.0),
        "news_flag": s.get("news_flag", False),
    })
    append_parquet({
        **s,
        "ts_jst": now_jst(),
        "gate": "passed" if ok else "blocked",
        "gate_meta": gate_meta,
        "run_id": run_id or new_run_id(),
    }, SIGNALS_LOG)
    return ok, gate_meta


def handle_before_execution(signal: Dict[str, Any], order_ctx: Dict[str, Any], gate_meta: Dict[str, Any], *, run_id: str | None = None) -> Tuple[bool, Dict[str, Any]]:
    """Apply TCA slippage guard before placing order. Logs rejection if any.

    order_ctx expects: spread, atr_1m, atr_15m, size, session, latency_ms, news_flag
    """
    # 1) Risk pre-entry check (daily cap / cooldown)
    ok_risk, reason = risk_check("pre_entry",
                                 equity_day_start_ccy=order_ctx.get("equity_day_start_ccy"),
                                 realized_pnl_today_ccy=order_ctx.get("realized_pnl_today_ccy"),
                                 current_losstreak=order_ctx.get("current_losstreak"))
    if not ok_risk:
        # map reason names to unified taxonomy
        reason_map = {
            "daily_loss_cap": "risk_daily_cap",
            "cooldown": "losstreak_cooldown",
        }
        unified_reason = reason_map.get(reason, reason)
        append_parquet({
            "ts_jst": now_jst(),
            "action": "reject",
            "reason": unified_reason,
            "run_id": run_id or new_run_id(),
            "variant": signal.get("variant"),
        }, ORDERS_LOG)
        return False, {"risk_reason": reason}

    # 2) TCA slippage guard
    slip = predict_slippage({
        "spread": order_ctx.get("spread"),
        "atr_1m": order_ctx.get("atr_1m"),
        "atr_15m": order_ctx.get("atr_15m"),
        "size": order_ctx.get("size"),
        "session": order_ctx.get("session"),
        "latency_ms": order_ctx.get("latency_ms"),
        "news_flag": order_ctx.get("news_flag", False),
    })
    # Compute EVs if TP/SL available in context; else fallback to provided ev_theory
    tp_pips = order_ctx.get("tp_pips") or order_ctx.get("TP_PIPS")
    sl_pips = order_ctx.get("sl_pips") or order_ctx.get("SL_PIPS")
    if tp_pips is not None and sl_pips is not None and signal.get("p_cal") is not None:
        p_cal = float(signal.get("p_cal"))
        ev_theory = p_cal * float(tp_pips) - (1.0 - p_cal) * float(sl_pips)
    else:
        ev_theory = float(signal.get("ev_theory", 0.0))
    ev_real = ev_theory - (float(order_ctx.get("spread", 0.0)) + abs(float(slip)))
    ev_pred = ev_real
    use_guard = bool(gate_meta.get("slippage_guard_ok", True))
    ev_min = float(gate_meta.get("ev_pred_min", 0.0))
    ok = (not use_guard) or (ev_real >= ev_min)
    if not ok:
        append_parquet({
            "ts_jst": now_jst(),
            "action": "reject",
            "reason": "ev_real_below_threshold",
            "slip_p50": slip,
            "ev_real_pips": ev_real,
            "ev_theory_pips": ev_theory,
            "ev_pred": ev_pred,
            "run_id": run_id or new_run_id(),
            "variant": signal.get("variant"),
        }, ORDERS_LOG)
    return ok, {"slip_p50": slip, "ev_pred": ev_pred, "ev_real": ev_real, "ev_theory": ev_theory}


def handle_after_fill(order: Dict[str, Any], trade: Dict[str, Any], *, run_id: str | None = None) -> None:
    """Append fill and trade info to logs."""
    append_parquet({**order, "ts_jst": now_jst(), "run_id": run_id or new_run_id()}, ORDERS_LOG)
    append_parquet({**trade, "run_id": run_id or new_run_id()}, TRADES_LOG)
    # update risk state with realized PnL if present
    try:
        pnl_ccy = trade.get("pnl_ccy")
        if pnl_ccy is None and "pnl_pips" in trade:
            # if only pips provided, record in pips as ccy-neutral (still usable for streaks)
            pnl_ccy = float(trade.get("pnl_pips") or 0.0)
        risk_on_close(pnl_ccy)  # type: ignore[arg-type]
    except Exception:
        pass
