import os
import pandas as pd

from inference_break import load_break_model, load_break_meta, predict_with_session_theta
from ml.time_consistency import build_features
from features_util import augment_features
from policy.gate import apply_final_gate


def test_e2e_smoke_local_data():
    # 1) load small local data slice (no network)
    data_csv = os.path.join("data", "USDJPY_15m.csv")
    assert os.path.exists(data_csv), "sample price csv not found"
    raw = pd.read_csv(data_csv)
    # Try to normalize columns
    cols = {c.lower(): c for c in raw.columns}
    for need in ["timestamp", "open", "high", "low", "close", "volume"]:
        assert any(k == need for k in (k.lower() for k in raw.columns)), f"missing col: {need}"
    # ensure dtype
    raw["timestamp"] = pd.to_datetime(raw[[c for c in raw.columns if c.lower()=="timestamp"][0]])
    raw = raw.sort_values("timestamp").tail(400).reset_index(drop=True)

    # 2) features
    feats_base = build_features(
        raw.rename(columns={c: c.lower() for c in raw.columns})[["timestamp","open","high","low","close","volume"]]
    )
    feats = augment_features(feats_base, raw.rename(columns={c: c.lower() for c in raw.columns}))
    assert not feats.empty

    # 3) model + meta
    model, use_cols = load_break_model()
    meta = load_break_meta()

    # 4) predict
    # align columns
    use_cols = list(use_cols)
    missing = [c for c in use_cols if c not in feats.columns]
    assert not missing, f"missing features: {missing[:5]}"
    df_pred = predict_with_session_theta(feats, model, use_cols, meta)

    # 5) apply gate (no streamlit)
    state = {
        "enable_trading": True,
        "auto_pause_on_drift": False,
        "drift_state": "normal",
        "apply_news_filter": False,
        "guard_state": {},
    }
    out = apply_final_gate(df_pred, None, state=state)
    assert {"trade_ok", "gate_reason"}.issubset(set(out.columns))
    # basic sanity
    assert out["proba"].between(0, 1).all()
