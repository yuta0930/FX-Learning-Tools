import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from config.loader import get_config
from src.monitoring.logs import plot_reliability

# 既存の学習ロジックを流用してバリデーション分割を構築
from ai_train_break import make_dataset, get_Xy
from label_break import BreakLabelConfig
from purged_cv import PurgedGroupTimeSeriesSplit, make_time_groups


def load_validation_split(use_cols_expected=None):
    cfg = get_config()
    data_path = os.path.join(cfg.paths.data_dir, f"{cfg.general.symbol}_{cfg.general.timeframe}.csv")
    raw = pd.read_csv(data_path)
    # BreakLabelConfig は既定で使用（必要なら config に昇格可能）
    lcfg = BreakLabelConfig()
    df = make_dataset(raw, horizon_bars=lcfg.H, buffer_ratio=0.0, label_config=lcfg)
    # グループ（1日）単位で PurgedGroupTimeSeriesSplit
    groups = make_time_groups(df["timestamp"], freq="D")
    n_splits = 5
    embargo_groups = 1
    cv = PurgedGroupTimeSeriesSplit(n_splits=n_splits, group_gap=0, embargo_groups=embargo_groups)
    # 最後のfoldを検証とする
    last = None
    for tr_idx, te_idx in cv.split(df, groups=groups):
        last = (tr_idx, te_idx)
    if last is None:
        raise RuntimeError("validation split could not be created")
    _, te_idx = last
    X_all, y_all, used_cols = get_Xy(df)
    # use_cols が指定されていれば順序を合わせる
    if use_cols_expected is not None and list(used_cols) != list(use_cols_expected):
        # 再抽出: used_cols -> DataFrame から expected の順に
        X_df = df[used_cols]
        # 期待列がすべて存在していることを確認
        missing = [c for c in use_cols_expected if c not in X_df.columns]
        if missing:
            raise RuntimeError(f"features missing for validation: {missing}")
        X_all = X_df[use_cols_expected].values.astype(float)
    X_val = X_all[te_idx]
    y_val = y_all[te_idx]
    # 欠損除外
    m = ~np.isnan(y_val)
    return X_val[m], y_val[m]


def main():
    cfg = get_config()
    models_dir = cfg.paths.models_dir
    base_path = os.path.join(models_dir, 'break_model.joblib')
    out_path = os.path.join(models_dir, 'break_model_calibrated.joblib')

    pkg = joblib.load(base_path)
    base_model = pkg['model']
    use_cols = pkg.get('use_cols') or pkg.get('Xcols')

    X_val, y_val = load_validation_split(use_cols_expected=use_cols)
    method = cfg.calibration.method if hasattr(cfg, 'calibration') else 'isotonic'
    cal = CalibratedClassifierCV(base_model, cv='prefit', method=method)
    cal.fit(X_val, y_val)

    pkg_out = {
        'model': cal,
        'use_cols': pkg.get('use_cols') or pkg.get('Xcols')
    }
    joblib.dump(pkg_out, out_path)

    # メトリクス（簡易）
    from sklearn.metrics import brier_score_loss
    p_raw = base_model.predict_proba(X_val)[:,1]
    p_cal = cal.predict_proba(X_val)[:,1]
    b_raw = brier_score_loss(y_val, p_raw)
    b_cal = brier_score_loss(y_val, p_cal)
    print(f"Brier raw={b_raw:.6f}  cal={b_cal:.6f}")

    try:
        fig = plot_reliability(y_val, p_cal, title='Reliability (Calibrated)')
        fig.savefig(os.path.join(cfg.paths.logs_dir, 'reliability_calibrated.png'), dpi=150)
    except Exception:
        pass

if __name__ == '__main__':
    main()
