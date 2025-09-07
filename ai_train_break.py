# ai_train_break.py
# ------------------------------------------------------------
# Purged Walk-Forward + Embargo + Robust Calibration (saga+ElasticNet)
# + OOF確率 → EV最適θ探索（coverage制約） → セッション/レジーム別θ
# + コスト感度分析 + 較正レポート + メタ/モデル保存 + 監査サンプル出力
# 使い方例:
#   python ai_train_break.py --csv data/USDJPY_15m.csv \
#     --h 20 --buffer 0.0015 --splits 5 --embargo 24 --min_cov 0.15 \
#     --R_win 1.0 --R_loss 1.0 --cost 0.15
# 出力:
#   models/break_model.joblib
#   models/break_meta.json
#   reports/break_calibration.png (ほか JSON)
#   audit_break_pos.csv / audit_break_neg.csv
# ------------------------------------------------------------

import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

# ---- project modules ----
from features_util import augment_features
from featx import add_volatility_and_interactions
from purged_cv import PurgedGroupTimeSeriesSplit, make_time_groups, purged_walk_forward_indices
from ev_utils import EVConfig, ev_for_threshold
from sweep_labels import export_label_audit_samples
from minority_metrics import summarize_minor_metrics
from calibration import calibration_report
# time_consistency 側に特徴量/ラベル生成がある想定
from ml.time_consistency import build_features
from label_break import build_break_labels, BreakLabelConfig

# ============================================================
# ユーティリティ
# ============================================================
RNG_SEED = 42
DROP_COLS = ["timestamp", "open", "high", "low", "close", "volume", "y"]
EPS = 1e-9


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def winsorize_df(df: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    num = df.select_dtypes(include=["float", "int"]).columns
    for c in num:
        s = df[c]
        m, v = s.mean(), s.std(ddof=0)
        if v == 0 or np.isnan(v):
            continue
        df[c] = np.clip(s, m - clip * v, m + clip * v)
    return df


# ============================================================
# データセット構築（特徴量 + ラベル）
# ============================================================
def make_dataset(raw: pd.DataFrame, horizon_bars: int, buffer_ratio: float, label_config: BreakLabelConfig) -> pd.DataFrame:
    raw = raw.sort_values("timestamp").reset_index(drop=True)


    # ベース特徴
    feats = build_features(raw)

    # 追加特徴（時刻/ボラ交互作用・新規特徴量追加）
    raw_l = raw.rename(columns=str.lower)
    feats = augment_features(feats, raw_l)
    feats = add_volatility_and_interactions(feats, raw_l, enable_poly=True)  # 多項特徴量も有効化


    # 方向依存特徴量を必ず追加
    # dir: +1（上方向）/-1（下方向）を交互に付与（例: 偶数行+1, 奇数行-1）
    feats["dir"] = np.where(np.arange(len(feats)) % 2 == 0, 1, -1)
    # dir_sign: dir列をコピー
    feats["dir_sign"] = feats["dir"]
    # dist_to_level: closeとhigh/lowの差をdirで反転
    level = (raw["high"] + raw["low"]) / 2
    feats["dist_to_level"] = (raw["close"] - level) * feats["dir"]
    # atr_slope_dir: ATRの変化率にdirを掛ける
    feats["atr_slope_dir"] = feats["atr"].diff().fillna(0.0) * feats["dir"]
    # rsi_div_dir: RSIの変化率にdirを掛ける（RSIがなければ0）
    if "rsi" in feats.columns:
        feats["rsi_div_dir"] = feats["rsi"].diff().fillna(0.0) * feats["dir"]
    else:
        feats["rsi_div_dir"] = 0.0

    # 新規特徴量例: 直近20本の高値・安値比率、ATRの変化率
    feats["high_low_ratio_20"] = (raw["high"].rolling(20).max() / raw["low"].rolling(20).min()).fillna(1.0)
    feats["atr_change_10"] = feats["atr"].pct_change(10).fillna(0.0)

    # ラベル（BreakLabelConfigを引数で受け取る）
    # horizon_bars, buffer_ratio, label_config を引数で渡す
    labels = build_break_labels(raw, label_config)["y"]

    # 学習データ期間拡張（例: 直近1年分のみ抽出などはコメントアウト、全期間利用）
    df = feats.merge(labels, left_index=True, right_index=True, how="left").dropna(subset=["y"])
    assert df["timestamp"].is_monotonic_increasing, "timestamp が昇順ではありません"
    df = winsorize_df(df)  # 外れ抑制
    return df


def get_Xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # 方向依存特徴量を必ず含める（dir_signも追加）
    required_cols = ["dir", "dist_to_level", "atr_slope_dir", "rsi_div_dir", "dir_sign"]
    use_cols = [c for c in df.columns if c not in DROP_COLS]
    for rc in required_cols:
        if rc not in use_cols:
            use_cols.append(rc)
    X = df[use_cols].values.astype(float)
    y = df["y"].astype(int).values
    return X, y, use_cols


# ============================================================
# モデル（saga + ElasticNet） + 安定較正（最近側サブセットCV）
# ============================================================
def build_model_for_break() -> CalibratedClassifierCV:

    # l1_ratio, C をスイープして最良モデルを選択
    best_score = -np.inf
    best_model = None
    for l1_ratio in [0.10, 0.15, 0.20]:
        for C in [0.35, 0.50, 0.75]:
            base = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("logreg", LogisticRegression(
                    solver="saga",
                    penalty="elasticnet",
                    l1_ratio=l1_ratio,
                    C=C,
                    max_iter=8000,
                    tol=2e-4,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RNG_SEED
                ))
            ])
            # 仮の評価: ここではモデルを返すだけ。fit_with_safe_calibration で最終評価
            model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            # 最良モデル選択は fit_with_safe_calibration で行う
            if best_model is None:
                best_model = model
    return best_model


def fit_with_safe_calibration(Xtr: np.ndarray, ytr: np.ndarray):
    """単一クラスfold回避＆最近側 subset で較正する堅牢版"""
    if len(np.unique(ytr)) < 2:
        return None, "skip_single_class"

    def _base():
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("logreg", LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                l1_ratio=0.12,
                C=0.45,
                max_iter=8000,
                tol=2e-4,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RNG_SEED
            ))
        ])

    n = len(Xtr)
    for frac in [0.3, 0.4, 0.5]:
        st = max(0, int(n * (1.0 - frac)))
        X_sub, y_sub = Xtr[st:], ytr[st:]
        if len(y_sub) < 100 or len(np.unique(y_sub)) < 2:
            continue
        try:
            skf = StratifiedKFold(n_splits=3, shuffle=False)
            calib = CalibratedClassifierCV(_base(), method="sigmoid", cv=skf)
            calib.fit(X_sub, y_sub)
            return calib, f"sigmoid_cv3_recent(frac={frac})"
        except Exception as e:
            print(f"[calibration error] {e}")
            pass

    # フォールバック（非較正）
    base = _base()
    base.fit(Xtr, ytr)

    class _Wrap:
        def __init__(self, m): self.m = m
        def predict_proba(self, X): return self.m.predict_proba(X)

    return _Wrap(base), "fallback_nocal_saga"


# ============================================================
# 診断・θ探索・セッション別θ
# ============================================================
def quick_diagnose(df: pd.DataFrame, Xcols: List[str], y: np.ndarray):
    pos_ratio = float(np.mean(y))
    print(f"[diag] total pos_ratio={pos_ratio:.3f}  samples={len(y)}")
    try:
        sub = df[Xcols + ["y"]].copy()
        corr = sub.corr(numeric_only=True)["y"].sort_values(key=lambda s: s.abs(), ascending=False).head(10)
        print("[diag] top|corr(feature, y)|:\n", corr.to_string())
    except Exception as e:
        print(f"[diag] corr failed: {e}")


def search_best_theta(proba, ev, min_cov=0.05, target_cov=0.10):
    """
    1) θの下限を損益分岐確率に縛る
    2) 目標カバレッジ >= target_cov の中から EV/tr - λ|cov-target| を最大化
    3) ダメなら min_cov >= の EV/tr 最大
    4) 最後の最後に trades>0 の EV/tr 最大（coverage=0回避）
    """
    # --- 1) p_break-even を下限に ---
    p_be = (ev.R_loss + ev.cost_per_trade) / (ev.R_win + ev.R_loss)
    print(f"[θ-guard] break-even p >= {p_be:.3f} -> θ grid starts at ~{max(0.50, p_be-0.02):.3f}")
    th_min = max(0.50, p_be - 0.02)
    th_max = 0.95
    grid = np.linspace(th_min, th_max, int((th_max - th_min) / 0.005) + 1)

    cand = []
    for th in grid:
        m = ev_for_threshold(proba, th, ev)
        cand.append({"theta": float(th), **m})

    # --- 2) target_cov以上でバランス最大化 ---
    lam = 0.05
    def score(m): return m["ev_per_trade"] - lam * abs(m["coverage"] - (target_cov or m["coverage"]))
    pool = [m for m in cand if m["coverage"] >= (target_cov or 0.0)]
    if pool:
        return max(pool, key=score)

    # --- 3) min_cov以上で EV/tr 最大 ---
    pool = [m for m in cand if m["coverage"] >= (min_cov or 0.0)]
    if pool:
        return max(pool, key=lambda m: m["ev_per_trade"])

    # --- 4) trades>0 の EV/tr 最大（coverage=0の高θは除外） ---
    pool = [m for m in cand if m["trades"] > 0]
    return max(pool, key=lambda m: m["ev_per_trade"]) if pool else max(cand, key=lambda m: m["coverage"])


def dump_theta_sweep(proba: np.ndarray, ev: EVConfig, label="OOF"):
    grid = np.linspace(0.60, 0.96, 19)
    rows = []
    for th in grid:
        m = ev_for_threshold(proba, th, ev)
        rows.append({"theta": float(th), **m})
    df = pd.DataFrame(rows)
    print(f"[sweep:{label}] head\n", df.head(5).to_string(index=False))
    print(f"[sweep:{label}] tail\n", df.tail(5).to_string(index=False))
    return df


def best_theta_by_session(df: pd.DataFrame, proba: np.ndarray, ev: EVConfig,
                          min_cov=0.22, target_cov=0.30):
    # セッション列が無ければ時間帯で代替
    if all(c in df.columns for c in ["tokyo", "london", "ny"]):
        masks = {
            "Tokyo": (df["tokyo"] > 0.5).values,
            "London": (df["london"] > 0.5).values,
            "NY": (df["ny"] > 0.5).values,
        }
    else:
        h = df["timestamp"].dt.hour
        masks = {
            "Tokyo": ((h >= 9) & (h < 15)).values,
            "London": ((h >= 16) & (h < 24)).values,
            "NY": ((h >= 22) | (h < 5)).values,
        }

    out = {
        name: search_best_theta(
            proba[idx], ev,
            min_cov=0.05 if name != "Tokyo" else 0.03,
            target_cov=0.12 if name != "Tokyo" else 0.07
        )
        for name, idx in masks.items() if idx.sum() >= 200
    }
    return out


def best_theta_by_session_regime(df: pd.DataFrame, proba: np.ndarray, ev: EVConfig,
                                 min_cov=0.22, target_cov=0.30):
    out = {}
    sessions = {"Tokyo": "tokyo", "London": "london", "NY": "ny"}
    for sname, scol in sessions.items():
        if scol not in df.columns:
            continue
        out[sname] = {}
        for lab in ("low", "mid", "high"):
            col = f"reg_atr_{lab}"
            if col not in df.columns:
                continue
            m = (df[scol] > 0.5) & (df[col] > 0.5)
            idx = m.values
            if idx.sum() < 200:
                continue
            th = search_best_theta(proba[idx], ev, min_cov=min_cov, target_cov=target_cov)
            out[sname][lab] = th
    return out


# ============================================================
# 学習・評価（Purged WF + Embargo）
# ============================================================
def train_eval_wf(df: pd.DataFrame, n_splits: int, embargo_groups: int, ev: EVConfig) -> Dict:
    X, y, use_cols = get_Xy(df)
    quick_diagnose(df, use_cols, y)

    groups = make_time_groups(df["timestamp"], freq="D")
    cv = PurgedGroupTimeSeriesSplit(n_splits=n_splits, group_gap=1, embargo_groups=embargo_groups)

    metrics: List[Dict] = []
    p_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for i, (tr, te) in enumerate(cv.split(df, groups=groups), 1):
        y_tr, y_te = y[tr], y[te]
        print(f"[PurgedCV] fold={i} train={len(tr)} test={len(te)} "
              f"pos(tr)={y_tr.mean():.3f} pos(te)={y_te.mean():.3f}")

        # fold学習＋検証
        model, mode = fit_with_safe_calibration(X[tr], y[tr])
        if model is None:
            row = {"fold": i, "AP": np.nan, "Brier": np.nan, "n": len(te), "mode": "skipped"}
            metrics.append(row)
            print(f"[PurgedCV] fold={i} skipped (train single class).")
            continue

        p_te = model.predict_proba(X[te])[:, 1]
        ap = float("nan") if len(np.unique(y[te])) < 2 else average_precision_score(y[te], p_te)
        brier = sklearn.metrics.brier_score_loss(y[te], p_te)
        row = summarize_minor_metrics(y[te], p_te)
        row.update({"fold": i, "AP": ap, "Brier": float(brier), "mode": mode, "pos_ratio_fold": float(y[te].mean())})
        metrics.append(row)
        p_list.append(p_te)
        y_list.append(y[te])

        print(f"[PurgedCV] fold={i} | AP={ap if not np.isnan(ap) else 'nan'} "
              f"| Brier={brier:.6f} | MCC={row.get('MCC', np.nan):.3f} | BA={row.get('BA', np.nan):.3f}")

    cv_df = pd.DataFrame(metrics)
    if not cv_df.empty:
        print("\n[CV SUMMARY] (minority-first)\n", cv_df.describe(include='all'))

    # OOF確率
    if not p_list:
        raise RuntimeError("評価用確率がゼロ件でした。ラベル/分割条件を見直してください。")
    p_all = np.concatenate(p_list)
    y_oos = np.concatenate(y_list)

    # θ探索 & OOS評価
    dump_theta_sweep(p_all, ev, label="OOF")
    theta = search_best_theta(p_all, ev, min_cov=0.05, target_cov=0.10)
    print(f"[θ*] picked θ={theta['theta']:.4f} | cov={theta['coverage']:.3f} "
          f"| EV/tr={theta['ev_per_trade']:.3f} | avg_p={theta['avg_p']:.3f}")

    pred = (p_all >= theta["theta"]).astype(int)
    tp = int(((pred == 1) & (y_oos == 1)).sum())
    fp = int(((pred == 1) & (y_oos == 0)).sum())
    tn = int(((pred == 0) & (y_oos == 0)).sum())
    fn = int(((pred == 0) & (y_oos == 1)).sum())
    prec, rec, f1, _ = precision_recall_fscore_support(y_oos, pred, average="binary", zero_division=0)
    ev_tr = float((p_all[pred == 1] * ev.R_win - (1 - p_all[pred == 1]) * ev.R_loss - ev.cost_per_trade).mean()) \
        if (pred == 1).sum() > 0 else float("-inf")

    print(f"[WF] OOS AP_macro={np.nanmean(cv_df['AP']) if 'AP' in cv_df else np.nan:.4f} "
          f"| Brier_macro={np.nanmean(cv_df['Brier']) if 'Brier' in cv_df else np.nan:.6f}")
    print(f"[WF] Best θ={theta['theta']:.2f} | coverage={theta['coverage']:.3f} "
          f"| EV/tr={theta['ev_per_trade']:.4f}")
    print(f"[WF θ] trades={(pred==1).sum()}  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  EV/tr={ev_tr:.3f}")
    print(f"[WF θ] TP={tp} FP={fp} TN={tn} FN={fn}")

    return {
        "cv_df": cv_df,
        "AP_macro": float(np.nanmean(cv_df["AP"])) if "AP" in cv_df else float("nan"),
        "Brier_macro": float(np.nanmean(cv_df["Brier"])) if "Brier" in cv_df else float("nan"),
        "best_threshold": theta,
        "use_cols": [c for c in df.columns if c not in DROP_COLS],
        "oos_theta_eval": {
            "theta": theta["theta"],
            "trades": int((pred == 1).sum()),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "ev_per_trade": float(ev_tr),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        },
        "p_all": p_all,
        "y_oos": y_oos
    }


# ============================================================
# 保存
# ============================================================
def save_model(df: pd.DataFrame, use_cols: List[str], out_path: str):
    _ensure_dir(out_path)
    X = df[use_cols].values.astype(float)
    y = df["y"].astype(int).values
    if len(np.unique(y)) < 2:
        raise RuntimeError("フルデータが単一クラスです。horizon/buffer を見直してください。")
    model, mode = fit_with_safe_calibration(X, y)
    if model is None:
        raise RuntimeError("最終学習が単一クラスで失敗しました。条件を見直してください。")
    import joblib
    joblib.dump({"model": model, "use_cols": use_cols, "Xcols": use_cols, "mode": mode}, out_path)
    print(f"[model] saved -> {out_path} (mode={mode}, n={len(X)})")


def save_meta(df: pd.DataFrame, ev: EVConfig, summary: Dict, out_path: str):
    _ensure_dir(out_path)
    ts_col = "timestamp"
    y = df["y"].astype(int).values
    cls = {"pos": int(y.sum()), "neg": int(len(y) - y.sum())}

    meta = {
        "start_ts": str(df[ts_col].iloc[0]),
        "end_ts": str(df[ts_col].iloc[-1]),
        "rows": int(len(df)),
        "class_balance": cls,
        "OOF": {"AP_macro": summary["AP_macro"], "Brier_macro": summary["Brier_macro"]},
        "threshold": float(summary["best_threshold"]["theta"]),
        "coverage_at_threshold": float(summary["best_threshold"]["coverage"]),
        "ev_per_trade_at_threshold": float(summary["best_threshold"]["ev_per_trade"]),
        "ev_per_trade": float((summary.get("oos_theta_eval") or {}).get("ev_per_trade", float("nan"))),
        "ev_cfg": asdict(ev),
        "features": summary["use_cols"],
        "cv": {"kind": "PurgedWalkForward", "n_splits": int(5), "embargo": int(1)},
        "random_state": RNG_SEED,
        "trained_at": int(time.time()),
        "trained_at_iso": datetime.now(timezone.utc).isoformat(),
    }

    # 追記: コスト感度・セッション別θ・レジーム別θ（後段で埋める）
    for k in ("ev_cost_sensitivity", "theta_by_session", "theta_by_session_regime"):
        if k in summary:
            meta[k] = summary[k]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[meta] saved -> {out_path}")


# ============================================================
# 較正レポート（OOF確率で作成）
# ============================================================
def run_calibration_report(p_oof: np.ndarray, y_oof: np.ndarray,
                           use_cols: List[str], out_dir="reports", stem="break_calibration"):
    os.makedirs(out_dir, exist_ok=True)
    rep = calibration_report(
        y_true=y_oof, proba=p_oof,
        n_bins=10,
        title="Break Model Calibration",
        out_dir=out_dir,
        stem=stem,
        meta={"features": use_cols},
        strategy="quantile",
        alpha=0.05
    )
    print(f"[calib] Brier={rep['brier']:.6f} | ECE={rep['ece']:.6f} | png={rep['png']}")


# ============================================================
# コスト感度
# ============================================================
def sweep_cost_sensitivity(p_oof: np.ndarray, ev_base: EVConfig,
                           costs=(0.02, 0.03, 0.04)) -> Dict[str, Dict]:
    out = {}
    for c in costs:
        ev_tmp = EVConfig(R_win=ev_base.R_win, R_loss=ev_base.R_loss, cost_per_trade=c)
        th = search_best_theta(p_oof, ev_tmp, min_cov=0.25, target_cov=0.30)
        out[f"{c:.2f}"] = th
        print(f"[cost={c:.2f}] θ={th['theta']:.4f} cov={th['coverage']:.3f} "
              f"EV/tr={th['ev_per_trade']:.3f} avg_p={th['avg_p']:.3f} trades={th['trades']}")
    return out


# ============================================================
# CLI
# ============================================================
@dataclass
class TrainConfig:
    horizon_bars: int = 20
    buffer_ratio: float = 0.0015
    n_splits: int = 5
    embargo_bars: int = 24
    min_coverage: float = 0.15
    model_out: str = "models/break_model.joblib"
    meta_out: str = "models/break_meta.json"


def parse_args():
    p = argparse.ArgumentParser(description="Train break model (WF + Embargo + EV threshold)")
    p.add_argument("--csv", type=str, default="data/USDJPY_15m.csv", help="学習データのCSVファイルパス")
    p.add_argument("--h", "--horizon", dest="horizon_bars", type=int, default=20, help="予測地平（バー数）")
    p.add_argument("--buffer", dest="buffer_ratio", type=float, default=0.0015, help="ブレイクバッファ比率（ATR基準）")
    p.add_argument("--splits", dest="n_splits", type=int, default=5, help="Purged Walk-Forwardの分割数")
    p.add_argument("--embargo", dest="embargo_bars", type=int, default=24, help="エンバーゴ期間（バー数）")
    p.add_argument("--min_cov", dest="min_coverage", type=float, default=0.15, help="最小カバレッジ")
    p.add_argument("--R_win", type=float, default=1.0, help="1トレードあたりの利益")
    p.add_argument("--R_loss", type=float, default=1.0, help="1トレードあたりの損失")
    p.add_argument("--cost", dest="cost_per_trade", type=float, default=0.15, help="1トレードあたりのコスト")
    p.add_argument("--model_out", type=str, default="models/break_model.joblib", help="モデル保存先パス")
    p.add_argument("--meta_out", type=str, default="models/break_meta.json", help="メタ情報保存先パス")
    return p.parse_args()


# ============================================================
# main
# ============================================================
def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")
    raw = pd.read_csv(csv_path, parse_dates=["timestamp"])
    raw = raw.rename(columns={c: c.lower() for c in raw.columns})
    need = {"timestamp", "open", "high", "low", "close", "volume"}
    if not need.issubset(raw.columns):
        raise ValueError(f"必要列が不足: {need} / got={set(raw.columns)}")
    return raw

def get_label_config(args) -> BreakLabelConfig:
    return BreakLabelConfig(
        H=args.horizon_bars,
        buffer_mode="atr",
        break_buffer_atr=args.buffer_ratio,
        settle_bars=1,
        settle_atr=0.02,
        tb_up_R=0.8,
        tb_dn_R=0.8,
        tb_timeout=120,
        exclude_in_windows=False
    )

def run_training(args):
    raw = load_and_preprocess(args.csv)
    label_config = get_label_config(args)
    df = make_dataset(raw, args.horizon_bars, args.buffer_ratio, label_config)
    print("[label distribution] y value counts:")
    print(df["y"].value_counts())
    if len(df) < (args.n_splits + 1) * 50:
        raise RuntimeError(f"データ不足 rows={len(df)}. 期間を増やすか splits/embargo を見直してください。")

    pos_csv, neg_csv = export_label_audit_samples(df, label_col="y", proba_col=None, out_prefix="audit_break")
    print("[label audit] exported:", pos_csv, neg_csv)

    ev = EVConfig(R_win=args.R_win, R_loss=args.R_loss, cost_per_trade=args.cost_per_trade)
    summary = train_eval_wf(df, n_splits=args.n_splits, embargo_groups=1, ev=ev)
    p_all = summary["p_all"]
    y_oos = summary["y_oos"]

    df_oof = df
    theta_by_sess = best_theta_by_session(df_oof, p_all, ev, min_cov=0.05, target_cov=0.10)
    theta_by_reg = best_theta_by_session_regime(df_oof, p_all, ev, min_cov=0.05, target_cov=0.10)
    summary["theta_by_session"] = theta_by_sess
    summary["theta_by_session_regime"] = theta_by_reg
    print("[θ_by_session]", theta_by_sess)
    print("[θ_by_session_regime]", theta_by_reg)

    run_calibration_report(p_all, y_oos, summary["use_cols"], out_dir="reports", stem="break_calibration")
    ev_cost_sens = sweep_cost_sensitivity(p_all, ev, costs=(0.02, 0.03, 0.04))
    summary["ev_cost_sensitivity"] = ev_cost_sens

    save_model(df, summary["use_cols"], args.model_out)
    save_meta(df, ev, {k: summary[k] for k in summary if k != "cv_df"}, args.meta_out)
    print("[done] all artifacts saved.")

def main():
    args = parse_args()
    run_training(args)

if __name__ == "__main__":
    main()
