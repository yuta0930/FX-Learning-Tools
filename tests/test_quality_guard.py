import types
import numpy as np
from ai_train_break import run_training, parse_args
import tempfile, json, os, pandas as pd

# We monkeypatch run_training dependencies lightly by creating a minimal CSV.

def _make_min_csv(path):
    # minimal columns required by load_and_preprocess + features + labels pipeline
    # For safety just create a few rows with dummy OHLCV and timestamp; y will be generated in later pipeline
    import datetime as dt
    rows = []
    now = pd.Timestamp.utcnow().floor('min')
    for i in range(300):
        ts = now + pd.Timedelta(minutes=15*i)
        rows.append([ts,1.0,1.1,0.9,1.05,1000])
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df.to_csv(path, index=False)

# This test only checks that quality guard does not crash when metrics are poor/high.
# Full integration of model training is heavy; this is a smoke style test.

def test_quality_guard_smoke(monkeypatch):
    tmp = tempfile.TemporaryDirectory()
    csv_path = os.path.join(tmp.name, 'data.csv')
    _make_min_csv(csv_path)

    # Patch train_eval_wf to return a crafted poor summary to trigger rejection
    from ai_train_break import save_model, save_meta
    def fake_train_eval_wf(df, n_splits, embargo_groups, ev):
        return {
            'AP_macro': 0.0,
            'Brier_macro': 0.99,
            'best_threshold': {'theta':0.9,'coverage':0.005,'ev_per_trade':-0.1},
            'use_cols': [c for c in df.columns if c not in ['y']],
            'oos_theta_eval': {'ev_per_trade': -0.1},
            'p_all': np.array([0.4,0.5,0.6]),
            'y_oos': np.array([0,1,0])
        }
    monkeypatch.setattr('ai_train_break.train_eval_wf', fake_train_eval_wf)
    # Patch heavy functions
    monkeypatch.setattr('ai_train_break.make_dataset', lambda raw, h, b, lc: raw.assign(y=0))
    monkeypatch.setattr('ai_train_break.export_label_audit_samples', lambda *a, **k: ("pos.csv","neg.csv"))
    monkeypatch.setattr('ai_train_break.best_theta_by_session', lambda *a, **k: {})
    monkeypatch.setattr('ai_train_break.best_theta_by_session_regime', lambda *a, **k: {})
    monkeypatch.setattr('ai_train_break.run_calibration_report', lambda *a, **k: None)
    monkeypatch.setattr('ai_train_break.sweep_cost_sensitivity', lambda *a, **k: {})
    monkeypatch.setattr('ai_train_break.save_model', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('should not be called when rejected')))  # Should not be called
    monkeypatch.setattr('ai_train_break.save_meta', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('should not be called when rejected')))

    args = types.SimpleNamespace(
        csv=csv_path, horizon_bars=20, buffer_ratio=0.0015, n_splits=3,
        embargo_bars=24, min_coverage=0.15, R_win=1.0, R_loss=1.0, cost_per_trade=0.15,
        model_out=os.path.join(tmp.name,'m.joblib'), meta_out=os.path.join(tmp.name,'m.json')
    )
    from ai_train_break import run_training
    run_training(args)
    # Expect a rejected json file
    assert os.path.exists(os.path.join(tmp.name,'m_rejected.json'))
    tmp.cleanup()
