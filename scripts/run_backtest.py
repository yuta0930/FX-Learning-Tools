# 最小の呼び出し例（ダミー）。実データと既存推論を流用して拡張してください。
import os
import json
import joblib
import pandas as pd
import numpy as np
import _bootstrap  # noqa: F401
from config.loader import get_config
from src.backtest.walkforward import WalkForward, WFConfig
from src.monitoring.metrics import brier_score, ece_score
from src.monitoring.logs import Logger, TradeLog
from inference_break import load_break_model, load_break_meta
from ml.time_consistency import build_features
from features_util import augment_features
from src.features.regime import RegimeClassifier, RegimeConfig
from src.execution.executor import ExecutionEngine, ExecutionConfig
from src.risk.risk_manager import RiskManager, RiskConfig
import matplotlib.pyplot as plt


def main():
    cfg = get_config()
    data_path = f"{cfg.paths.data_dir}/{cfg.general.symbol}_{cfg.general.timeframe}.csv"
    raw = pd.read_csv(data_path)
    wf = WalkForward(WFConfig(cfg.backtest.wf_window_days, cfg.backtest.wf_step_days))
    splits = wf.split(raw)
    print(f"WF splits: {len(splits)} windows")
    if not splits:
        return

    # 全期間で特徴量を前計算
    raw_l = raw.rename(columns=str.lower)
    feats = build_features(raw_l)
    feats = augment_features(feats, raw_l)

    # モデルロード（base必須、calibratedは任意）
    base_pkg = joblib.load(os.path.join(cfg.paths.models_dir, 'break_model.joblib'))
    base_model = base_pkg['model']
    use_cols = base_pkg.get('use_cols') or base_pkg.get('Xcols')
    cal_model = None
    cal_path = os.path.join(cfg.paths.models_dir, 'break_model_calibrated.joblib')
    if os.path.exists(cal_path):
        cal_pkg = joblib.load(cal_path)
        cal_model = cal_pkg['model']

    # レジームクラス
    rcfg = RegimeConfig(
        method=cfg.regime.method,
        k=int(cfg.regime.k),
        atr_window=int(cfg.regime.atr_window),
        rv_window=int(cfg.regime.rv_window),
        session_dummies=bool(cfg.regime.session_dummies),
    )
    reg = RegimeClassifier(rcfg)

    # θ（メタ）
    try:
        meta = load_break_meta(os.path.join(cfg.paths.models_dir, 'break_meta.json'))
        theta = float(meta.get('threshold', 0.6))
    except Exception:
        theta = 0.6

    # リスク/執行設定
    ee_cfg = ExecutionConfig(
        slippage_alpha=float(cfg.execution.slippage_alpha),
        slippage_beta=float(cfg.execution.slippage_beta),
        max_retry=int(cfg.execution.max_retry),
        cool_down_bars=int(cfg.execution.cool_down_bars),
        allowed_regimes=list(cfg.execution.allowed_regimes),
    )
    rm_cfg = RiskConfig(
        daily_loss_limit_pct=float(cfg.risk.daily_loss_limit_pct),
        max_drawdown_pct=float(cfg.risk.max_drawdown_pct),
        kelly_cap=float(cfg.risk.kelly_cap),
        base_risk_per_trade_pct=float(cfg.risk.base_risk_per_trade_pct),
        reduce_on_dd=bool(cfg.risk.reduce_on_dd),
    )

    # 集計用
    agg_base_y = []
    agg_base_p = []
    agg_cal_y = []
    agg_cal_p = []
    pnl_base = []
    pnl_cal = []
    eq_base = []
    eq_cal = []
    # 方向別の集計
    pnl_base_long = []
    pnl_base_short = []
    pnl_cal_long = []
    pnl_cal_short = []

    # ロガー
    logger = Logger(cfg.paths.logs_dir, parquet=bool(cfg.logging.parquet), jsonl=bool(cfg.logging.jsonl))

    # 全WF窓を評価
    per_window_metrics = []  # WFごとの確率メトリクスを保持
    for w in splits:
        te_mask = (pd.to_datetime(raw['timestamp'])>=pd.to_datetime(w['test_start'])) & (pd.to_datetime(raw['timestamp'])<pd.to_datetime(w['test_end']))
        te_idx = np.where(te_mask.values)[0]
        if te_idx.size == 0:
            continue
        te_df = raw.iloc[te_idx].copy()
        X_te = feats.iloc[te_idx][use_cols].values.astype(float)
        y_te = te_df['y'].values.astype(float) if 'y' in te_df.columns else None
        p_base = base_model.predict_proba(X_te)[:,1]
        p_cal = cal_model.predict_proba(X_te)[:,1] if cal_model is not None else None

        # レジーム/市場状態
        te_reg = reg.transform(te_df)
        regimes = te_reg['regime_name'].values
        # リスク・執行用の指標
        hl_over_close = ((te_df['high']-te_df['low']).abs()/te_df['close']).fillna(0.0).values
        rv = np.log(te_df['close']).diff().rolling(30, min_periods=5).std().fillna(0.0).values
        atr = te_reg['atr'].fillna(method='ffill').fillna(0.0).values
        close = te_df['close'].values

        # 検証データがある場合のみ集計・シミュレーション
        if y_te is not None:
            agg_base_y.append(y_te); agg_base_p.append(p_base)
            if p_cal is not None:
                agg_cal_y.append(y_te); agg_cal_p.append(p_cal)

            # この窓の確率メトリクスを記録
            try:
                win_base = {
                    'Brier': float(brier_score(y_te, p_base)),
                    'ECE': float(ece_score(y_te, p_base)),
                    'Hit': float(np.mean(y_te == 1.0)),
                    'Samples': int(len(y_te)),
                }
                win_cal = None
                if p_cal is not None:
                    win_cal = {
                        'Brier': float(brier_score(y_te, p_cal)),
                        'ECE': float(ece_score(y_te, p_cal)),
                        'Hit': float(np.mean(y_te == 1.0)),
                        'Samples': int(len(y_te)),
                    }
                per_window_metrics.append({
                    'start': str(w['test_start']),
                    'end': str(w['test_end']),
                    'base': win_base,
                    'cal': win_cal,
                })
            except Exception:
                # 1窓の失敗は無視して続行
                pass

            # 窓ごとにエンジン/リスクをリセット
            for label, p_vec, pnl_list, eq_list, pnl_long, pnl_short in (
                ('base', p_base, pnl_base, eq_base, pnl_base_long, pnl_base_short),
                ('cal',  p_cal,  pnl_cal,  eq_cal,  pnl_cal_long,  pnl_cal_short),
            ):
                if p_vec is None:
                    continue
                # 再現性のためにseed固定のRNGを使用
                seed = int(getattr(cfg.general, 'seed', 42))
                ee = ExecutionEngine(ee_cfg, rng=np.random.default_rng(seed))
                rm = RiskManager(rm_cfg)
                eq = 0.0
                for i in range(len(p_vec)):
                    ee.tick()
                    if not ee.can_trade(str(regimes[i])) or rm.should_halt_new():
                        continue
                    p = float(p_vec[i])
                    # 双方向の閾値: p>=theta でロング、p<=1-theta でショート
                    side = None
                    p_win = None  # 選択方向の勝率
                    if p >= theta:
                        side = 'buy'
                        p_win = p
                    elif p <= (1.0 - theta):
                        side = 'sell'
                        p_win = 1.0 - p
                    if side is None:
                        continue

                    # ATRを距離スケールとして利用（割合ベース）
                    atr_abs = float(max(atr[i], 1e-9))
                    px = float(max(close[i], 1e-9))
                    atr_dist = float(min(1.0, atr_abs / px))  # 0〜1にクリップ
                    R_multiple = 1.0  # シンプルに 1R 固定
                    f = rm.position_size_pct(p=p_win, atr_distance=atr_dist, R=R_multiple)
                    # スリッページは常にコスト（割合）として控除
                    slip_abs = abs(ee.apply_slippage(side, px, float(hl_over_close[i]), float(rv[i])) - px)
                    slip_frac = float(min(0.5, slip_abs / px))

                    # アウトカム: y==1 は"上方向"の成功、y==0 は"下方向"の成功とみなす
                    is_win = (y_te[i] == 1.0 and side == 'buy') or (y_te[i] != 1.0 and side == 'sell')
                    pnl = f * (R_multiple if is_win else -R_multiple) - f * slip_frac

                    rm.update_equity(pnl)
                    ee.on_fill()
                    pnl_list.append(pnl)
                    (pnl_long if side=='buy' else pnl_short).append(pnl)
                    eq += pnl
                    eq_list.append(eq)
                    # ログ 1トレード=1レコード
                    logger.log_trade(TradeLog(
                        timestamp=str(te_df['timestamp'].iloc[i]),
                        regime=str(regimes[i]),
                        proba_raw=float(p_vec[i]),
                        proba_cal=float(p_vec[i]) if label=='cal' else float('nan'),
                        side=side,
                        size=float(f),
                        entry=float('nan'),
                        exit=float('nan'),
                        stop=float('nan'),
                        take=float('nan'),
                        spread_proxy=float(hl_over_close[i]),
                        slippage=float(slip_frac),
                        pnl=float(pnl),
                        dd=float(rm.drawdown_pct()),
                        flags={'stage': label}
                    ))

    # メトリクス集計
    def summarize(y_list, p_list):
        if not y_list or not p_list:
            return {}
        y = np.concatenate(y_list)
        p = np.concatenate(p_list)
        return {
            'Brier': brier_score(y, p),
            'ECE': ece_score(y, p),
            'Hit': float(np.mean(y==1.0)),
            'Samples': int(len(y))
        }

    def trade_stats(pnl_series):
        if not pnl_series:
            return {}
        pnl = np.asarray(pnl_series, float)
        pos = pnl[pnl>0].sum(); neg = -pnl[pnl<0].sum()
        eq = np.cumsum(pnl)
        dd = np.maximum.accumulate(eq) - eq
        return {
            'trades': int(np.count_nonzero(pnl!=0.0)),
            'PF': float(pos/neg) if neg>0 else float('inf'),
            'MaxDD': float(np.max(dd)) if len(dd)>0 else 0.0,
            'Sharpe': float(np.mean(pnl)/ (np.std(pnl)+1e-9) * np.sqrt(max(1,len(pnl)))) if len(pnl)>1 else 0.0,
            'PnL': float(np.sum(pnl)),
        }

    base_prob = summarize(agg_base_y, agg_base_p)
    cal_prob  = summarize(agg_cal_y,  agg_cal_p) if agg_cal_p else {}
    base_tr   = trade_stats(pnl_base)
    cal_tr    = trade_stats(pnl_cal) if pnl_cal else {}

    print("== Probability metrics (WF aggregated) ==")
    print("Base:", base_prob)
    if cal_prob:
        print("Calib:", cal_prob)
    print("== Trading metrics (WF aggregated) ==")
    # 方向別
    base_by_side = {'long': trade_stats(pnl_base_long), 'short': trade_stats(pnl_base_short)}
    cal_by_side = {'long': trade_stats(pnl_cal_long), 'short': trade_stats(pnl_cal_short)} if pnl_cal else {}
    print("Base:", base_tr, "by_side=", base_by_side)
    if cal_tr:
        print("Calib:", cal_tr, "by_side=", cal_by_side)

    # ログflush（parquet/jsonl）
    logger.flush()
    # レポート（JSON + 図）
    try:
        os.makedirs(cfg.paths.reports_dir, exist_ok=True)
        report = {
            'prob': {'base': base_prob, 'cal': cal_prob},
            'prob_by_window': per_window_metrics,
            'trade': {
                'base': base_tr,
                'cal': cal_tr,
                'base_by_side': base_by_side,
                'cal_by_side': cal_by_side if cal_tr else {}
            }
        }
        with open(os.path.join(cfg.paths.reports_dir, 'wf_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        if eq_base:
            plt.figure(figsize=(6,3))
            plt.plot(eq_base, label='Base')
            if eq_cal:
                plt.plot(eq_cal, label='Calib')
            plt.legend(); plt.title('Equity (WF aggregated, simple)'); plt.tight_layout()
            plt.savefig(os.path.join(cfg.paths.reports_dir, 'equity_wf.png'), dpi=150)
            plt.close()
    except Exception as e:
        print('[report][warn]', e)

if __name__ == '__main__':
    main()
