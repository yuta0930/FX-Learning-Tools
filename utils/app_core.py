"""
UIに依存しない小さなコア関数群。テストや他モジュールから安全に利用できる。
"""
from __future__ import annotations

import json
import numpy as np
from typing import Tuple, Optional
import pandas as pd


def _load_and_validate_baseline(
    meta_path: str = "models/break_meta.json",
    calib_path: str = "reports/break_calibration.json",
    *,
    # デフォルト閾値: 2 以上あれば利用可能とする（テスト期待に合わせる）。
    # 1 サンプルだけの場合は分布推定として不十分とみなし None 扱い。
    min_samples: int = 2,
    warn_on_clip: bool = True,
) -> tuple[float, Optional[np.ndarray], list[str]]:
    """
    ベースライン確率分布を読み込み検証する（UI非依存）。

    Returns: (baseline_proba: float, baseline_probs: np.ndarray | None, warnings: list[str])
    """
    warns: list[str] = []
    base_p = 0.5
    probs_arr = None

    # meta
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
        base_p = float(meta_json.get("baseline_proba", 0.5))
        if not (0.0 <= base_p <= 1.0):
            warns.append(f"baseline_proba 異常値={base_p:.4f} -> 0.5 にリセット")
            base_p = 0.5
    except FileNotFoundError:
        warns.append(f"metaファイル未検出: {meta_path}")
    except json.JSONDecodeError as e:
        warns.append(f"meta JSON 解析失敗: {e}")
    except Exception as e:
        warns.append(f"meta読込失敗: {e}")

    # calibration
    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_json = json.load(f)
        raw = calib_json.get("prob_mean") or calib_json.get("calibration", {}).get("prob_mean")
        if raw is not None:
            probs_arr = np.asarray(raw, dtype=float)
    except FileNotFoundError:
        warns.append(f"calibrationファイル未検出: {calib_path}")
    except json.JSONDecodeError as e:
        warns.append(f"calibration JSON 解析失敗: {e}")
    except Exception as e:
        warns.append(f"calibration読込失敗: {e}")

    # validate
    if probs_arr is not None:
        if probs_arr.size == 0:
            warns.append("baseline_probs が空 -> 利用不可 (PSI skip)")
            probs_arr = None
        else:
            nan_mask = ~np.isfinite(probs_arr)
            if nan_mask.any():
                ratio = nan_mask.mean()
                warns.append(f"baseline_probs NaN/inf {ratio:.1%} -> 除去")
                probs_arr = probs_arr[~nan_mask]
            if probs_arr.size == 0:
                warns.append("baseline_probs 除去後に空 -> 利用不可")
                probs_arr = None
            else:
                if not ((probs_arr >= 0).all() and (probs_arr <= 1).all()):
                    if warn_on_clip:
                        warns.append("baseline_probs に 0-1 範囲外 -> クリップ")
                    probs_arr = np.clip(probs_arr, 0, 1)
                if probs_arr.size < min_samples:
                    warns.append(f"baseline_probs サンプル不足 {probs_arr.size} < {min_samples} -> 不使用")
                    probs_arr = None

    return base_p, probs_arr, warns


def compute_drift_score(
    dm: dict,
    w: dict | None = None,
    caps: dict | None = None,
) -> float:
    """
    ドリフト指標 dm={'psi','kl','js','hellinger'} を 0-1 に正規化して合算し 0-1 を返す（UI非依存）。
    """
    w = w or {'psi':0.5,'kl':0.2,'js':0.2,'h':0.1}
    caps = caps or {'psi':0.5,'kl':0.5,'js':0.5,'h':1.0}
    psi = float(dm.get('psi', float('nan')))
    kl  = float(dm.get('kl',  float('nan')))
    js  = float(dm.get('js',  float('nan')))
    h   = float(dm.get('hellinger', float('nan')))

    def nz(x: float) -> float:
        return x if np.isfinite(x) and x >= 0 else 0.0

    psi_n = min(1.0, nz(psi) / max(1e-9, caps.get('psi',0.5)))
    kl_n  = min(1.0, nz(kl)  / max(1e-9, caps.get('kl',0.5)))
    js_n  = min(1.0, nz(js)  / max(1e-9, caps.get('js',0.5)))
    h_n   = min(1.0, nz(h)   / max(1e-9, caps.get('h', 1.0)))
    score = (psi_n * w.get('psi',0.5) +
             kl_n  * w.get('kl', 0.2) +
             js_n  * w.get('js', 0.2) +
             h_n   * w.get('h',  0.1))
    wsum = sum([w.get('psi',0.5), w.get('kl',0.2), w.get('js',0.2), w.get('h',0.1)]) or 1.0
    return float(np.clip(score / wsum, 0.0, 1.0))


def compute_final_theta_components(
    base_theta: float,
    session_theta: float,
    drift_bump: float,
    news_bump: float,
    theta_min: float,
    theta_max: float,
):
    """
    最終θの純粋計算コンポーネント。UI非依存。
    優先順位: base vs session は高い方、そこにドリフト/ニュースのバンプを加算し[min,max]でクリップ。

    Returns: (theta_final, breakdown_dict)
    """
    # 安全ガード: 範囲が逆転している場合は入れ替える
    tmin = float(theta_min)
    tmax = float(theta_max)
    if tmin > tmax:
        tmin, tmax = tmax, tmin

    base_after_session = max(float(base_theta), float(session_theta))
    b_session = base_after_session - float(base_theta)
    b_drift = float(drift_bump) if np.isfinite(drift_bump) else 0.0
    b_news = float(news_bump) if np.isfinite(news_bump) else 0.0
    theta_final = base_after_session + b_drift + b_news
    theta_final = float(np.clip(theta_final, tmin, tmax))
    breakdown = {
        'base': float(base_theta),
        'session_used': float(session_theta),
        'session_bump': float(b_session),
        'drift_bump': float(b_drift),
        'news_bump': float(b_news),
        'min': float(tmin),
        'max': float(tmax),
    }
    return theta_final, breakdown


# --- 追加: UIに依存しない最終θ計算（ニュース/セッション込み） ---
def _pick_session_from_ts(ts: pd.Timestamp) -> Optional[str]:
    try:
        h = int(ts.hour)
    except Exception:
        return None
    if 9 <= h < 15:
        return "Tokyo"
    if 16 <= h < 24:
        return "London"
    if h >= 22 or h < 5:
        return "NY"
    return None


def _pick_theta_from_meta(meta: dict, ts: pd.Timestamp) -> float:
    sess = _pick_session_from_ts(ts)
    tbs = (meta or {}).get("theta_by_session", {})
    if sess and sess in tbs and isinstance(tbs[sess], dict) and "theta" in tbs[sess]:
        try:
            return float(tbs[sess]["theta"])
        except Exception:
            pass
    try:
        return float((meta or {}).get("threshold", 0.93))
    except Exception:
        return 0.93


def _is_in_news_window_pure(
    ts: pd.Timestamp,
    windows_df: Optional[pd.DataFrame],
    *,
    mode_label: str = "重要度別（赤影と同じ）",
    news_win_minutes: int = 30,
    imp_min: int = 3,
) -> bool:
    if windows_df is None or len(windows_df) == 0:
        return False
    # 既存のデフォルトモードと同じ: start/end に包含されるか
    if mode_label == "重要度別（赤影と同じ）":
        try:
            return bool(((windows_df["start"] <= ts) & (ts <= windows_df["end"])).any())
        except Exception:
            return False
    # 他モードは未対応: ここではFalse（将来拡張余地）
    return False


def compute_final_theta_for_time_pure(
    ts: pd.Timestamp,
    meta: dict,
    windows_df: Optional[pd.DataFrame],
    settings: dict,
) -> tuple[float, dict]:
    """
    最終θをUI非依存で計算。
    settings 例:
        {
          'theta_min': 0.40,
          'theta_max': 0.85,
          'theta_adaptive': 0.60,  # なければ 'theta_base' を参照
          'theta_base': 0.60,
          'theta_drift_bump_active': True/False,
          'theta_bump_drift': 0.03,
          'use_soft_suppress': True/False,
          'theta_bump_in_news': 0.03,
          'news_win': 30,
          'news_imp_min': 3,
          'news_filter_mode': "重要度別（赤影と同じ）",
        }
    """
    theta_min = float(settings.get('theta_min', 0.40))
    theta_max = float(settings.get('theta_max', 0.85))
    base_theta = float(settings.get('theta_adaptive', settings.get('theta_base', 0.60)))

    try:
        session_theta = _pick_theta_from_meta(meta, ts)
    except Exception:
        session_theta = base_theta

    in_news = _is_in_news_window_pure(
        ts,
        windows_df,
        mode_label=str(settings.get('news_filter_mode', "重要度別（赤影と同じ）")),
        news_win_minutes=int(settings.get('news_win', 30)),
        imp_min=int(settings.get('news_imp_min', 3)),
    )
    soft_active = bool(in_news and settings.get('use_soft_suppress', False))

    drift_bump = float(settings.get('theta_bump_drift', 0.03)) if settings.get('theta_drift_bump_active', False) else 0.0
    news_bump = float(settings.get('theta_bump_in_news', 0.03)) if soft_active else 0.0

    theta_final, breakdown = compute_final_theta_components(
        base_theta=base_theta,
        session_theta=session_theta,
        drift_bump=drift_bump,
        news_bump=news_bump,
        theta_min=theta_min,
        theta_max=theta_max,
    )
    breakdown['in_news'] = bool(in_news)
    return float(theta_final), breakdown


# --- ニュース: 重要度別ウィンドウ生成（純粋関数） ---
def build_event_windows_pure(
    events_df: Optional[pd.DataFrame],
    imp_threshold: int,
    mapping: dict[int, int],
) -> pd.DataFrame:
    """
    イベントDFから [start,end] の抑制ウィンドウを生成する（UI非依存）。
    必須カラム: time, importance（title は任意）
    返却カラム: start, end, importance, title
    """
    cols = ["start", "end", "importance", "title"]
    if events_df is None or len(events_df) == 0:
        return pd.DataFrame(columns=cols)
    if not {"time", "importance"}.issubset(set(events_df.columns)):
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    for _, r in events_df.iterrows():
        try:
            imp = int(r.get("importance", 0))
        except Exception:
            imp = 0
        if imp < int(imp_threshold):
            continue
        minutes = int(mapping.get(imp, 0))
        t = r.get("time")
        if pd.isna(t):
            continue
        start = t - pd.Timedelta(minutes=minutes)
        end = t + pd.Timedelta(minutes=minutes)
        rows.append({
            "start": start,
            "end": end,
            "importance": imp,
            "title": r.get("title", ""),
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    windows = pd.DataFrame(rows)
    return windows.sort_values("start").reset_index(drop=True)
