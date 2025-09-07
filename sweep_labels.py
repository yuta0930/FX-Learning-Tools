import pandas as pd
import numpy as np

def _detect_time_col(df: pd.DataFrame) -> str:
    for c in ["time", "timestamp", "dt", "date", "datetime"]:
        if c in df.columns:
            return c
    raise KeyError("time/datetime系の列が見つかりません（time/timestamp/dt/date）")

def export_label_audit_samples(
    df: pd.DataFrame,
    label_col: str = "y",
    proba_col: str | None = None,
    n_pos: int = 120,
    n_neg: int = 120,
    session_cols: tuple[str, ...] = ("tokyo", "london", "ny"),
    vola_col: str | None = "atr14_norm_v",
    out_prefix: str = "label_audit",
    random_state: int = 42,
):
    """
    人手監査用に y=1 / y=0 を層化サンプリングしてCSV出力。
    - セッション（列が存在すれば）× ボラ（存在すれば四分位）で層化
    - proba_col があれば予測確率でソート（低確信/高確信を混ぜるため頭/尻から均等抽出）
    出力:
      {out_prefix}_pos.csv, {out_prefix}_neg.csv
    """
    rng = np.random.default_rng(random_state)
    tcol = _detect_time_col(df)
    base = df.copy()

    # 層キー作成
    sess_key = (
        base[list(session_cols)].astype(int).astype(str).agg("".join, axis=1)
        if all(c in base.columns for c in session_cols) else pd.Series("na", index=base.index)
    )
    if vola_col and vola_col in base.columns:
        vola_bin = pd.qcut(base[vola_col].rank(method="first"), q=4, labels=["Q1","Q2","Q3","Q4"])
    else:
        vola_bin = pd.Series("Q?", index=base.index)

    base["_layer"] = sess_key + "|" + vola_bin.astype(str)
    assert label_col in base.columns, f"{label_col} が存在しません"
    if proba_col and proba_col in base.columns:
        # 確率の両端から半分ずつ抜く（誤ラベル検知に効く）
        base["_proba_rank"] = base[proba_col].rank(pct=True)
    else:
        base["_proba_rank"] = 0.5

    out = {}
    for yval, n_samp, tag in [(1, n_pos, "pos"), (0, n_neg, "neg")]:
        cand = base[base[label_col] == yval].copy()
        samples = []
        for layer, g in cand.groupby("_layer"):
            k = max(1, int(n_samp / max(1, cand["_layer"].nunique())))
            if proba_col and proba_col in g.columns:
                g = g.sort_values("_proba_rank")
                k2 = k // 2
                head = g.head(k2)
                tail = g.tail(k - k2)
                take = pd.concat([head, tail]).sample(frac=1.0, random_state=random_state)
            else:
                take = g.sample(n=min(k, len(g)), random_state=random_state)
            samples.append(take)
        out[tag] = pd.concat(samples).sort_values(tcol).reset_index(drop=True)

        # 最低限使う列だけ落とす（人手監査に十分な情報）
        keep_cols = [c for c in [tcol, label_col, proba_col, vola_col] if c and c in base.columns]
        # 参考に特徴を数列だけ追加（過剰なら削ってOK）
        num_feats = base.select_dtypes(include=[np.number]).columns.tolist()
        # 重複列回避
        cols = list(dict.fromkeys(keep_cols + ["_layer"] + num_feats[:10]))
        out[tag][cols].to_csv(f"{out_prefix}_{tag}.csv", index=False)

    return f"{out_prefix}_pos.csv", f"{out_prefix}_neg.csv"
# sweep_labels.py（新規）
import itertools, json, numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss

from ml.time_consistency import build_features, build_break_labels, BreakLabelConfig
# 既存のものを流用
from features_util import augment_features
from cv_utils import purged_walk_forward_indices
from ev_utils import EVConfig, ev_for_threshold

def get_Xy(df):
    drop = ['timestamp','open','high','low','close','volume','y']
    cols = [c for c in df.columns if c not in drop]
    X = df[cols].values; y = df["y"].astype(int).values
    return X,y,cols

def build_model():
    base = LogisticRegression(max_iter=300, class_weight="balanced")
    return CalibratedClassifierCV(base, method="isotonic", cv=3)

def oof_proba_WF(X,y,splits):
    p = np.full(len(y), np.nan)
    for tr,te in splits:
        mdl = build_model(); mdl.fit(X[tr],y[tr])
        p[te] = mdl.predict_proba(X[te])[:,1]
    m = ~np.isnan(p)
    return p[m], y[m]

def search_best_theta(proba, ev, min_cov=0.25, target_cov=0.30):
    grid = np.linspace(0.55, 0.98, 88)
    best = {"theta":0.5,"ev_per_trade":-1e9,"trades":0,"coverage":0.0,"avg_p":0.0}
    def _try(min_required):
        nonlocal best
        for th in grid:
            m = ev_for_threshold(proba, th, ev)
            if m["coverage"]>=min_required and m["ev_per_trade"]>best["ev_per_trade"]:
                best = m
    if target_cov:
        _try(target_cov)
        if best["ev_per_trade"]<=-1e8: _try(min_cov)
    else:
        _try(min_cov)
    return best

def run_sweep(raw_csv="data/USDJPY_15m.csv", out_csv="reports/label_sweep.csv"):
    raw = pd.read_csv(raw_csv, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    base_feats = build_features(raw)
    raw_l = raw.rename(columns=str.lower)
    base_feats = augment_features(base_feats, raw_l)

    results=[]
    for h, buf, cost in itertools.product([8,10,12,16],[0.0005,0.0008,0.0010],[0.02,0.03]):
        labels = build_break_labels(raw, BreakLabelConfig(horizon_bars=h, buffer_ratio=buf)).rename("y")
        df = base_feats.merge(labels, left_index=True, right_index=True, how="left").dropna(subset=["y"])
        X,y,cols = get_Xy(df)
        splits = purged_walk_forward_indices(len(df), n_splits=5, embargo=12)
        p_oof, y_oof = oof_proba_WF(X,y,splits)
        ap = average_precision_score(y_oof, p_oof); brier = brier_score_loss(y_oof, p_oof)
        ev = EVConfig(R_win=1.0, R_loss=1.0, cost_per_trade=cost)
        theta = search_best_theta(p_oof, ev, min_cov=0.25, target_cov=0.30)
        results.append({
            "H":h, "buffer":buf, "cost":cost,
            "AP":float(ap), "Brier":float(brier),
            "theta":float(theta["theta"]), "coverage":float(theta["coverage"]),
            "EV_per_trade":float(theta["ev_per_trade"])
        })
        print(f"[{h},{buf},{cost}] AP={ap:.4f} Brier={brier:.4f} θ={theta['theta']:.3f} cov={theta['coverage']:.3f} EV/tr={theta['ev_per_trade']:.3f}")

    import os; os.makedirs("reports", exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("saved ->", out_csv)

if __name__ == "__main__":
    run_sweep()
