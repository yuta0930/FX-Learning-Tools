# Upgrade: Drift/Calibration/TCA/A-B/Dashboard Integration

This document summarizes the minimal-invasion upgrade that introduces:
- Drift monitoring (PSI/ECE) with gate adjustments
- Online calibration (Isotonic/Platt) and artifacts
- TCA (slippage modeling) and EV guard
- A/B gating manager with gradual promotion
- Lightweight Streamlit dashboard

## What was added
- configs/: gate.yml, ab.yml, drift.yml, tca.yml, dashboard.yml
- app/utils/: timeutil.py, io.py, ids.py, stats.py
- app/calibration/online_calibrator.py
- app/drift/drift_monitor.py
- app/tca/slippage_model.py, app/tca/tca_runner.py
- app/ab/ab_manager.py
- app/risk/risk_manager.py (persisted daily loss cap and cooldown)
- app/dashboard/dashboard.py
- components.py (hook functions)
- initialize.py (ensure_dirs)
- constants.py (paths/timezone)
- scripts/: run_weekly_jobs.py, backfill_logs.py
- tests/: test_new_*.py (smoke/unit tests for new modules)

## 2025-11-11 Update (Break Model & Features Alignment)
- Added level-relative features (`lvl_k1_up/dn`, `lvl_k2_up/dn`, `lvl_near_cnt`, `lvl_near_flag`, `lvl_span_atr`) unified between training and inference via `ml/time_consistency.build_features`.
- Fixed timestamp tz normalization mismatch (aware vs naive) that previously caused silent loss of `lvl_*` columns during merges.
- Introduced FAST training mode in `ai_train_break.py` (`--fast`): skips calibration search and uses lighter LBFGS L2 logistic for rapid iteration (pickle-safe).
- Reworked `fit_with_safe_calibration` to return picklable estimators directly (removed local `_Wrap` class) eliminating PicklingError on save.
- Added calibration report JSON/PNG emission during normal mode training; observed Brier/ECE improvement (FAST Brier≈0.2407 → Calibrated Brier≈0.2103, ECE≈0.0445).
- Updated README with lvl_* feature documentation and FAST mode usage command.

### Impact
- Training/inference feature parity restored (no hidden drops).
- Faster experiment loop (FAST) without sacrificing downstream calibration capability.
- More reliable artifact persistence (pickle compatibility) and history tracking (`reports/model_history.csv`).

### Follow-up Suggestions
- Optionally schedule periodic normal-mode recalibration (e.g., weekly) while using FAST for daily smoke retrains.
- Extend level-relative features with recent-touch time decay or density volatility normalization if EV gains plateau.
- Capture per-feature SHAP or permutation importance to quantify contribution of new lvl_* features.

## Minimal edits
- app.py: call initialize.ensure_dirs() once at startup (safe side effects only)

## Config toggles
- Turn features ON/OFF in configs/*.yml (see default values in each file).
   - gate.yml:
      - gate.enabled: true|false
      - gate.daily_loss_cap_pct: float (default 0.03)
      - gate.cooldown_on_losstreak: int (default 3)
      - gate.cooldown_minutes: int (default 45)
      - gate.use_slippage_guard: true|false
      - gate.ev_pred_min: float
   - drift.yml: thresholds for PSI/ECE and actions
   - ab.yml: allocation, variants, evaluate/promote policy
   - tca.yml: model type, features, min_samples

## Log schema (JST)
- data/logs/signals.parquet: [ts_jst, setup_id, features(json), p_raw, p_cal, quality, atr_15m, session, spread, news_flag, gate_decision, variant, run_id]
- data/logs/orders.parquet:  [ts_jst, order_id, setup_id, side, size, price_req, price_fill, slip_pips, spread, latency_ms, broker, run_id]
- data/logs/trades.parquet:  [ts_jst_open, ts_jst_close, order_id, pnl_pips, pnl_ccy, tp_hit, sl_hit, hold_secs, r_multiple, run_id]
- data/logs/tca_features.parquet: [ts_jst, spread, atr_1m, atr_15m, size, session, slip_pips, run_id]

## Hook integration (overview)
- components.py:
   - calibrate_proba(p_raw) -> p_cal (identity if calibrator missing)
   - assign_ab_variant(setup_id) -> "A"|"B"
   - gate_decision(signal_row) -> (ok, meta)  // includes slippage guard flags and ATR multipliers by variant
   - predict_slippage(feats_row) -> slip_pips_pred (p50/p90 by config)
   - risk_check("pre_entry", equity_day_start_ccy, realized_pnl_today_ccy, current_losstreak) -> (ok, reason)
   - risk_on_close(pnl_ccy): persist state update
- app/hooks_runner.py:
   - handle_signal(...): post-signal hooks + append to signals.parquet
   - handle_before_execution(...): risk_check -> TCA EV guard; logs rejection to orders.parquet
   - handle_after_fill(...): append fill/trade logs and risk_on_close

## How to run
1) Install deps (requirements.txt updated with statsmodels).
2) Start monitoring dashboard (optional):
   - streamlit run app/dashboard/dashboard.py
3) Weekly maintenance:
   - python scripts/run_weekly_jobs.py

## Rollback
- Remove the new directories (app/, configs/, artifacts/) and files listed above.
- Revert the one-line insertion in app.py (ensure_dirs call).

## Notes
- Symlink for calibration/current.joblib falls back to file copy on Windows.
- All new code includes docstrings and type hints; defaults are safe (halt on exceptions).
- Risk state persists to artifacts/risk/state_{yyyymmdd}.json for process restarts.
 - FAST モード利用時は未較正確率のため、本番前に `scripts/calibrate_break.py` で較正を推奨。