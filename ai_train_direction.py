# ai_train_direction.py — Up/Down 二系統モデル + Purged WF + EV Threshold
import os, json, joblib, numpy as np, pandas as pd, hashlib, time
from datetime import datetime, timezone
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _safe_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic replace

def build_and_save_meta(
    cfg,
    df,                               # 学習に使った行（time昇順）
    features,                          # 使用特徴量名 list[str]
    model_obj=None,                    # joblib保存したモデルオブジェクト（任意）
    symbol="USDJPY=X",
    interval="15m",
    period="60d",
    cv_kind="PurgedWalkForward",
    cv_n_splits=3,
    embargo=12,
    oof_auc=float("nan"),
    oof_ap=float("nan"),
    oof_brier=float("nan"),
    threshold=0.50,
    coverage_at_threshold=0.0,
    ev_per_trade=0.0,
    ev_cfg={"rr": 1.0, "cost_ratio": 0.03},
    calibrated=True,
    calibration_method="isotonic",
    meta_name="break_meta.json"
):
    # 期間情報
    ts_col = "timestamp" if "timestamp" in df.columns else df.index.name or "index"
    start_ts = str(df.iloc[0][ts_col]) if ts_col in df.columns else str(df.index[0])
    end_ts   = str(df.iloc[-1][ts_col]) if ts_col in df.columns else str(df.index[-1])

    # クラス比
    y = df["y"].astype(int).values if "y" in df.columns else None
    n_pos = int(y.sum()) if y is not None else None
    n_neg = int(len(df) - n_pos) if y is not None else None

    # コード/モデルのハッシュ（任意：あるファイルを基準に）
    try:
        with open(__file__, "rb") as f:
            code_hash = _sha256_bytes(f.read())
    except Exception:
        code_hash = None

    model_hash = None
    if model_obj is not None:
        try:
            import joblib, io
            buf = io.BytesIO()
            joblib.dump(model_obj, buf)
            model_hash = _sha256_bytes(buf.getvalue())
        except Exception:
            pass

    meta = {
        # ラベル条件（あなたの現状を維持＋明示）
        "horizon_bars": int(cfg.horizon_bars),
        "buffer_ratio": float(cfg.buffer_ratio),

        # データ条件
        "symbol": symbol,
        "interval": interval,
        "period": period,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "train_rows": int(len(df)),
        "class_balance": {"pos": n_pos, "neg": n_neg},

        # 検証条件
        "cv": {
            "kind": cv_kind,
            "n_splits": int(cv_n_splits),
            "embargo": int(embargo),
        },
        "OOF": {
            "AUC": None if (oof_auc != oof_auc) else float(oof_auc),          # NaN対策
            "AP":  None if (oof_ap  != oof_ap)  else float(oof_ap),
            "Brier": None if (oof_brier != oof_brier) else float(oof_brier),
        },

        # 閾値・期待値
        "threshold": float(threshold),
        "coverage_at_threshold": float(coverage_at_threshold),
        "ev_per_trade_at_threshold": float(ev_per_trade),
        "ev_cfg": {"rr": float(ev_cfg["rr"]), "cost_ratio": float(ev_cfg["cost_ratio"])} ,

        # モデル条件
        "model": {
            "class": getattr(getattr(model_obj, "__class__", None), "__name__", None),
            "features": list(features),
            "calibrated": bool(calibrated),
            "calibration_method": calibration_method,
            "model_hash": model_hash,
        },

        # ランダム性・再現
        "random_state": getattr(cfg, "random_state", None),
        "code_hash": code_hash,
        "trained_at": int(time.time()),
        "trained_at_iso": datetime.now(timezone.utc).isoformat(),

        # あなたの既存フィールドも保持
        "ev": getattr(cfg, "ev", None).__dict__ if hasattr(cfg, "ev") else None,
        "up_summary": globals().get("sum_up"),
        "down_summary": globals().get("sum_dn"),
    }

    # 保存
    path = os.path.join("models", meta_name)
    _safe_write_json(path, meta)
    print(f"[meta] saved -> {path}")
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss

from ml.time_consistency import build_features, BreakLabelConfig
from direction_labels import build_direction_labels, DirLabelConfig

# ===== Config =====
@dataclass
class TrainConfig:
    horizon_bars: int = 12
    buffer_ratio: float = 0.0005
    n_splits: int = 5
    embargo_bars: int = 12
    random_state: int = 42
    model_up: str = "break_up_model.joblib"
    model_down: str = "break_down_model.joblib"
    meta_name: str = "break_direction_meta.json"

@dataclass
class EVConfig:
    R_win: float = 1.0     # 勝ちR（TP）
    R_loss: float = 1.0    # 負けR（SL）
    cost_per_trade: float = 0.02  # 1トレードあたりのコスト（R換算）

# ===== Utils =====
def purged_walk_forward_indices(n: int, n_splits: int, embargo: int):
    if n_splits < 2: raise ValueError("n_splits>=2")
    test_size = n // (n_splits + 1)
    idx = np.arange(n)
    splits = []
    for k in range(1, n_splits + 1):
        t0 = k * test_size
        t1 = min(n, t0 + test_size)
        tr_end = max(0, t0 - embargo)
        tr_idx, te_idx = idx[:tr_end], idx[t0:t1]
        if len(tr_idx) and len(te_idx):
            splits.append((tr_idx, te_idx))
    if not splits: raise RuntimeError("データ不足でWF不可")
    return splits

def build_model():
    base = LogisticRegression(max_iter=300)
    return CalibratedClassifierCV(base, method="isotonic", cv=3)

def get_Xy(df: pd.DataFrame, target: str):
    drop_cols = ['timestamp','open','high','low','close','volume','y_up','y_down']
    use_cols = [c for c in df.columns if c not in drop_cols]
    X = df[use_cols].values
    y = df[target].astype(int).values
    return X, y, use_cols

def ev_for_threshold(proba: np.ndarray, theta: float, ev: EVConfig):
    sel = proba >= theta
    if sel.sum()==0:
        return {"theta":theta, "trades":0, "ev_per_trade":-np.inf, "coverage":0.0, "avg_p":0.0}
    p = proba[sel]
    evp = (p*ev.R_win) - ((1-p)*ev.R_loss) - ev.cost_per_trade
    return {"theta":theta, "trades":int(sel.sum()), "ev_per_trade":float(evp.mean()),
            "coverage": float(sel.mean()), "avg_p": float(p.mean())}

def search_best_theta(proba: np.ndarray, ev: EVConfig, min_cov=0.01):
    best = {"theta":0.5,"ev_per_trade":-1e9,"trades":0,"coverage":0.0,"avg_p":0.0}
    for th in np.linspace(0.5, 0.99, 100):
        m = ev_for_threshold(proba, th, ev)
        if m["coverage"]>=min_cov and m["ev_per_trade"]>best["ev_per_trade"]:
            best = m
    return best

# ===== Pipeline =====
def make_dataset(df_raw: pd.DataFrame, cfg: TrainConfig):
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)
    feats = build_features(df_raw)  # 未来を見ない特徴量
    lbls = build_direction_labels(df_raw, DirLabelConfig(cfg.horizon_bars, cfg.buffer_ratio))
    df = feats.merge(lbls, left_index=True, right_index=True, how='left').dropna(subset=['y_up','y_down'])
    assert df['timestamp'].is_monotonic_increasing
    return df

def wf_eval_one_target(df: pd.DataFrame, cfg: TrainConfig, ev: EVConfig, target: str):
    X, y, use_cols = get_Xy(df, target)
    splits = purged_walk_forward_indices(len(df), cfg.n_splits, cfg.embargo_bars)
    metrics, probas = [], []
    for i, (tr, te) in enumerate(splits, 1):
        mdl = build_model()
        mdl.fit(X[tr], y[tr])
        p = mdl.predict_proba(X[te])[:,1]
        ap = average_precision_score(y[te], p)
        brier = brier_score_loss(y[te], p)
        metrics.append({"split":i,"AP":float(ap),"Brier":float(brier),"n_test":int(len(te))})
        probas.append(p)
        print(f"[{target}] Split{i} AP={ap:.4f} Brier={brier:.4f} n={len(te)}")
    p_all = np.concatenate(probas)
    theta = search_best_theta(p_all, ev)
    ap_macro = float(np.mean([m["AP"] for m in metrics]))
    brier_macro = float(np.mean([m["Brier"] for m in metrics]))
    summary = {"AP_macro":ap_macro,"Brier_macro":brier_macro,"best_threshold":theta,"splits":metrics}
    print(f"[{target}] OOS AP_macro={ap_macro:.4f} Brier_macro={brier_macro:.4f} Bestθ={theta['theta']:.2f} EV/tr={theta['ev_per_trade']:.4f}")
    return summary, use_cols

def train_full_and_save(df: pd.DataFrame, cfg: TrainConfig, use_cols_up: List[str], use_cols_dn: List[str]):
    os.makedirs("models", exist_ok=True)
    # up
    X_up, y_up, _ = get_Xy(df, "y_up")
    m_up = build_model(); m_up.fit(X_up, y_up)
    joblib.dump({"model":m_up, "use_cols":use_cols_up}, os.path.join("models", cfg.model_up))
    # down
    X_dn, y_dn, _ = get_Xy(df, "y_down")
    m_dn = build_model(); m_dn.fit(X_dn, y_dn)
    joblib.dump({"model":m_dn, "use_cols":use_cols_dn}, os.path.join("models", cfg.model_down))

def main():
    # データ読み込み（列: timestamp, open, high, low, close, volume）
    df_raw = pd.read_csv("data/USDJPY_15m.csv", parse_dates=['timestamp'])
    cfg = TrainConfig()
    ev  = EVConfig(R_win=1.0, R_loss=1.0, cost_per_trade=0.02)

    df = make_dataset(df_raw, cfg)

    sum_up, use_up = wf_eval_one_target(df, cfg, ev, "y_up")
    sum_dn, use_dn = wf_eval_one_target(df, cfg, ev, "y_down")

    # メタ情報保存
    build_and_save_meta(
        cfg=cfg,
        df=df,
        features=use_up + use_dn,
        model_obj=None,
        symbol="USDJPY=X",
        interval="15m",
        period="60d",
        cv_kind="PurgedWalkForward",
        cv_n_splits=cfg.n_splits,
        embargo=cfg.embargo_bars,
        oof_auc=float('nan'),
        oof_ap=(sum_up["AP_macro"]+sum_dn["AP_macro"])/2 if ("AP_macro" in sum_up and "AP_macro" in sum_dn) else float('nan'),
        oof_brier=(sum_up["Brier_macro"]+sum_dn["Brier_macro"])/2 if ("Brier_macro" in sum_up and "Brier_macro" in sum_dn) else float('nan'),
        threshold=(sum_up["best_threshold"]["theta"]+sum_dn["best_threshold"]["theta"])/2 if ("best_threshold" in sum_up and "best_threshold" in sum_dn) else 0.5,
        coverage_at_threshold=(sum_up["best_threshold"]["coverage"]+sum_dn["best_threshold"]["coverage"])/2 if ("best_threshold" in sum_up and "best_threshold" in sum_dn) else 0.0,
        ev_per_trade=(sum_up["best_threshold"]["ev_per_trade"]+sum_dn["best_threshold"]["ev_per_trade"])/2 if ("best_threshold" in sum_up and "best_threshold" in sum_dn) else 0.0,
        ev_cfg={"rr": ev.R_win, "cost_ratio": ev.cost_per_trade},
        calibrated=True,
        calibration_method="isotonic",
        meta_name=cfg.meta_name
    )

    train_full_and_save(df, cfg, use_up, use_dn)
    print("Saved:", cfg.model_up, cfg.model_down, cfg.meta_name)

if __name__ == "__main__":
    main()
