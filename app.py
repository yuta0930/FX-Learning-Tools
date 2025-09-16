# 先に主要ライブラリを読み込む（上部で使用されるため）
import numpy as np
import pandas as pd
import streamlit as st

# === 共通処理関数 ===
def update_prob_buffer(prob_df):
    curr_probs = []
    if prob_df is not None:
        for _, r in prob_df.iterrows():
            for key in ["P_up", "P_dn"]:
                val = r.get(key, None)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    curr_probs.append(float(val))
    if curr_probs:
        st.session_state.prob_buffer.extend(curr_probs)
        st.session_state.prob_buffer = st.session_state.prob_buffer[-3000:]
    return curr_probs

def calc_psi_and_exrate(curr_probs, baseline_probs, theta_up, theta_dn):
    """PSI + 追加ドリフト指標(KL/JS/Hellinger) と閾値超過率をまとめて計算。

    Returns:
        psi_val, severity, exceed_rate, kl, js, hellinger
    """
    psi_val, sev, ex_rate = float('nan'), "n/a", float('nan')
    kl = js = hell = float('nan')
    if baseline_probs is not None and len(st.session_state.prob_buffer) >= 200:
        curr = np.array(st.session_state.prob_buffer[-1000:], dtype=float)
        try:
            psi_val, edges = compute_psi(baseline_probs, curr, edges=None)
            sev = psi_severity(psi_val)
            # 共有ビン edges を drift_summary 内で再利用させるため baseline/curr をそのまま渡す
            from monitoring import drift_summary  # 局所 import (循環回避安全)
            ds = drift_summary(baseline_probs, curr)
            kl = ds.get('kl', float('nan'))
            js = ds.get('js', float('nan'))
            hell = ds.get('hellinger', float('nan'))
        except Exception as e:
            print(f"ドリフト指標計算エラー: {e}")
        try:
            theta_rep = float(np.median([theta_up, theta_dn]))
            ex_rate = threshold_exceed_rate(np.array(curr_probs, float), theta_rep)
        except Exception as e:
            print(f"θ超過率計算エラー: {e}")
    return psi_val, sev, ex_rate, kl, js, hell
# （削除）重複していた __main__ テストブロックは冗長のため整理しました。
from monitoring import compute_psi, psi_severity, threshold_exceed_rate, healthcheck, safe_call, drift_summary
## --- 発注の最終ゲートに enable_trading を反映 ---
# pred_df, windows_dfが揃ったタイミングで以下を必ず通す
# pred_df: [timestamp, proba, theta, signal, ...]
# windows_df: [start, end]（JST）
# 例:
# in_news = is_in_any_window(pred_df["timestamp"], windows_df[["start","end"]])
# pred_df["trade_ok"] = (pred_df["signal"] == 1) & (~in_news)
# ↓ pred_df生成後、trade_ok列ができた直後に以下を追加してください
# pred_df["trade_ok"] = pred_df["trade_ok"] & st.session_state.enable_trading
#
# live_rows = pred_df.loc[pred_df["trade_ok"]]
# if live_rows.empty:
#     st.info("現在、発注条件を満たすシグナルはありません（EVゲート/ニュース抑制/θ適用後）。")
# else:
#     # live_rows を発注フック/通知へ
#     pass  # ここに発注処理を記述


from build_level_break_prob_table import build_level_break_prob_table
st.set_page_config(page_title="FX 自動ライン描画 - 完全版", page_icon="📈", layout="wide")
from inference_break import load_break_meta
from config.loader import get_config
from risk_guard import make_guard_from_config
import math

# --- ATR 計算ユーティリティ（単純版）---
def compute_latest_atr(price_df, period: int = 14):
    try:
        import pandas as pd  # 局所 import （存在しない場合は上位で失敗しているはず）
        req = {"high","low","close"}
        cols_lower = {c.lower() for c in price_df.columns}
        if not req.issubset(cols_lower):
            return float('nan')
        df = price_df.copy()
        ren = {}
        for c in df.columns:
            lc = c.lower()
            if lc in req:
                ren[c]=lc
        if ren:
            df = df.rename(columns=ren)
        hl = df['high'] - df['low']
        pc = df['close'].shift(1)
        h_pc = (df['high'] - pc).abs()
        l_pc = (df['low'] - pc).abs()
        import pandas as _pd
        tr = _pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=period).mean()
        val = atr.iloc[-1]
        return float(val) if np.isfinite(val) else float('nan')
    except Exception:
        return float('nan')

# --- ベースライン確率の読み込み（初期化 + 検証関数） ---
import json
import numpy as np
from functools import lru_cache

def _load_and_validate_baseline(meta_path: str = "models/break_meta.json",
                                 calib_path: str = "reports/break_calibration.json",
                                 *,
                                 min_samples: int = 5,
                                 warn_on_clip: bool = True):
    """ベースライン確率分布を読み込み検証するユーティリティ。

    読み込みソース:
      1. メタ JSON (`baseline_proba` フィールド: 閾値比較などに使う代表値)
      2. 較正 JSON (`prob_mean` 配列: 校正カーブ上の代表確率群) -> PSI / drift 監視用の近似分布として利用

    検証ルール:
      - NaN 削除（比率を警告）
      - 範囲外値は 0-1 にクリップ（オプション警告）
      - サンプル数閾値 (min_samples) 未満なら不安定警告し None 扱い

    Returns:
      (baseline_proba: float, baseline_probs: np.ndarray | None, warnings: list[str])

    実装メモ:
      - エラーは握りつぶしつつ warning に集約（UI レイヤで st.warning 表示）
      - 将来的に dataclass 返却へ拡張しやすいようタプル形式を維持
    """
    warns: list[str] = []
    base_p = 0.5
    probs_arr = None
    # --- meta 読み込み ---
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
        # fallback を 0.5 に固定（バイナリ無情報基準）
        base_p = float(meta_json.get("baseline_proba", 0.5))
        if not (0.0 <= base_p <= 1.0):  # 異常値はクリップせず警告し 0.5 に戻す
            warns.append(f"baseline_proba 異常値={base_p:.4f} -> 0.5 にリセット")
            base_p = 0.5
    except FileNotFoundError:
        warns.append(f"metaファイル未検出: {meta_path}")
    except json.JSONDecodeError as e:
        warns.append(f"meta JSON 解析失敗: {e}")
    except Exception as e:
        warns.append(f"meta読込失敗: {e}")

    # --- calibration 読み込み ---
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

    # --- 検証 ---
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

baseline_proba, baseline_probs, _baseline_warns = _load_and_validate_baseline()
if '_baseline_warns' in globals() and _baseline_warns:
    for w in _baseline_warns:
        st.warning(f"[baseline] {w}")

# === 推奨行動（意思決定ポリシー）関連 ===
from decision_policy import DecisionParams, EVConfig, recommend_action
from is_in_any_window import is_in_any_window  # 既に利用例あり

# ---- EV Gate: 初期化（最初の1回だけ） ----
if "enable_trading" not in st.session_state:
    try:
        meta = load_break_meta("models/break_meta.json")
        ev = float(meta.get("ev_per_trade", 0.0))
    except Exception:
        ev = 0.0  # 読み込み失敗＝安全側でOFF
    st.session_state.enable_trading = (ev > 0)

# ---- Risk Guard 初期化（最初の1回だけ） ----
if "trade_guard" not in st.session_state:
    try:
        cfg = get_config().risk_guard
        st.session_state.trade_guard = make_guard_from_config(cfg)
    except Exception as e:
        st.session_state.trade_guard = None
        st.warning(f"Risk guard 初期化失敗: {e}")

# ---- UI表示：以降はユーザー操作で上書き可能 ----
col1, col2 = st.columns([1,1])
with col1:
    enable_trading = st.toggle(
        "運用モード（EVゲート）",
    key="enable_trading"
    )
with col2:
    try:
        meta = load_break_meta("models/break_meta.json")
        ev = float(meta.get("ev_per_trade", float("nan")))
    except Exception:
        ev = float("nan")
    st.metric("EV per trade", f"{ev:.4f}" if ev == ev else "N/A")  # NaN対応
with st.sidebar.expander("🛡 Risk Guard", expanded=False):
    tg = st.session_state.get("trade_guard")
    if tg is None:
        st.write("(未初期化)")
    else:
        s = tg.state()
        st.write(f"日次Trades: {s['day_trades']} (max {s['max_day_trades']})")
        st.write(f"連敗: {s['consecutive_losses']} (limit {s['loss_cooldown_after']})")
        st.write(f"Cooldown: {'ON' if s['in_cooldown'] else 'OFF'} / 残 {s['cooldown_remaining_min']}m")
        if s.get('session_trades'):
            st.write("Session trades:")
            for k,v in s['session_trades'].items():
                st.write(f"- {k}: {v} / {s['max_session_trades']}")
        # 自動ATR計算（dfがあれば）
        latest_atr_disp = "n/a"
        _df_candidate = globals().get('df', None)
        if _df_candidate is not None:
            try:
                import pandas as _pd
                if isinstance(_df_candidate, _pd.DataFrame) and not _df_candidate.empty:
                    atr_period = s.get('atr_period', 14)
                    lat = compute_latest_atr(_df_candidate.tail(500), period=int(atr_period))
                    if np.isfinite(lat):
                        tg.record_atr(lat)
                        latest_atr_disp = f"{lat:.5f}"
            except Exception as e:
                st.write(f"ATR計算失敗: {e}")
        st.write(f"最新ATR(period={s.get('atr_period',14)}): {latest_atr_disp}")
        col_w, col_l = st.columns(2)
        with col_w:
            if st.button("Win記録", key="rg_mark_win"):
                tg.register_trade(result_pips=+1, ts=pd.Timestamp.utcnow(), session=None)
        with col_l:
            if st.button("Loss記録", key="rg_mark_loss"):
                tg.register_trade(result_pips=-1, ts=pd.Timestamp.utcnow(), session=None)
        s2 = tg.state()
        st.caption(f"更新後: 日次 {s2['day_trades']} / 連敗 {s2['consecutive_losses']}")
        # 詳細スナップショット（入れ子のエクスパンダは禁止のためコンテナで表示）
        with st.container():
            st.caption("詳細スナップショット")
            try:
                if hasattr(tg, 'snapshot') and callable(getattr(tg, 'snapshot')):
                    snap = tg.snapshot()
                else:
                    snap = tg.state()
                st.json(snap)
            except Exception as e:
                st.write(f"snapshot/state の取得に失敗: {e}")
            cols = st.columns(2)
            with cols[0]:
                if st.button("日次カウンタをリセット", key="rg_reset_day"):
                    try:
                        if hasattr(tg, 'reset_day') and callable(getattr(tg, 'reset_day')):
                            tg.reset_day()
                        else:
                            # 後方互換: 利用可能ならカウンタを0へ
                            st.warning("reset_day 未対応のため簡易リセットは省略しました。")
                        st.toast("日次カウンタをリセットしました", icon="♻️")
                    except Exception as e:
                        st.error(f"日次リセット失敗: {e}")
            with cols[1]:
                if st.button("スナップショット再取得", key="rg_resnap"):
                    try:
                        _ = tg.snapshot() if hasattr(tg, 'snapshot') and callable(getattr(tg, 'snapshot')) else tg.state()
                        st.toast("スナップショット更新", icon="🔄")
                    except Exception as e:
                        st.error(f"再取得失敗: {e}")

# --- ニュースキャッシュ メタ ---
with st.sidebar.expander("📰 ニュースキャッシュ", expanded=False):
    ns = st.session_state.get('news_stats', {'hits':0,'miss':0,'size':0})
    st.write(f"LRUサイズ: {ns.get('size',0)}  /  Hit: {ns.get('hits',0)}  Miss: {ns.get('miss',0)}")
    if st.button("キャッシュをクリア", key="clear_news_cache"):
        st.session_state.pop('news_win_cache', None)
        st.session_state['news_stats'] = {'hits':0,'miss':0,'size':0}
        st.toast("ニュースキャッシュをクリアしました", icon="🧹")
# --- セッション別カバレッジ確認用（UI/検証タブ等で利用） ---
# pred_dfに[session, signal]列がある前提
# 例: cov_by_sess = pred_df.groupby("session")["signal"].mean()
# 0.25〜0.35に収まっているかを確認し、外れる場合は次回学習でtarget_covを微調整
# --- ニュース抑制ウィンドウを注文ゲートに反映 ---
from is_in_any_window import is_in_any_window

# pred_df, windows_dfが揃ったタイミングで以下を必ず通す
# pred_df: [timestamp, proba, theta, signal, ...]
# windows_df: [start, end]（JST）
# 例:
# in_news = is_in_any_window(pred_df["timestamp"], windows_df[["start","end"]])
# pred_df["trade_ok"] = (pred_df["signal"] == 1) & (~in_news)
from inference_break import load_break_model, load_break_meta, predict_with_session_theta
from ev_cache import get_ev_tables

import streamlit as st

# === app.py に追加（import の下あたり） ===
import pandas as pd

# 学習時と同じ関数を使う
from ml.time_consistency import build_features
from features_util import augment_features  # 学習側と同一実装（循環 import 回避）

# === 最終θと注文ゲート適用のヘルパー ===
def get_final_theta(ts: pd.Timestamp | None = None, windows_df: pd.DataFrame | None = None, meta: dict | None = None) -> float:
    """既に計算済みの最終θがあればそれを返し、なければその場で合成する。"""
    th = st.session_state.get('theta_final')
    if isinstance(th, (int, float)) and np.isfinite(th):
        return float(th)
    # フォールバック: 現在時刻 / 画面上の windows / meta を用いて計算
    ts = ts or (pd.Timestamp.now(tz=JST))
    meta = meta or st.session_state.get('break_meta', {})
    windows_df = windows_df if windows_df is not None else st.session_state.get('windows_df', pd.DataFrame())
    try:
        th2, _ = compute_final_theta_for_time(ts, meta, windows_df)
        st.session_state['theta_final'] = th2
        return float(th2)
    except Exception:
        # 最低限、adaptive→base の順
        return float(st.session_state.get('theta_adaptive', st.session_state.get('theta_base', 0.60)))

def apply_final_gate(pred_df: pd.DataFrame,
                     windows_df: pd.DataFrame,
                     *,
                     signal_col: str = 'signal',
                     ts_col: str = 'timestamp',
                     add_columns: bool = True) -> pd.DataFrame:
    """注文ゲートを一括適用し、trade_ok 列を付与する。
    - 運用モード（enable_trading）
    - Drift 自動停止（alert かつ auto_pause）
    - Guard（in_cooldown のとき停止）
    - ニュース（ハード抑制のときに停止）
    ソフト抑制は最終θに織り込む方針のため、ここではブロックしない。
    """
    if not isinstance(pred_df, pd.DataFrame) or pred_df.empty:
        return pred_df
    df_out = pred_df.copy()
    # ニュース窓（ハード抑制）
    apply_news_filter = bool(st.session_state.get('apply_news_filter', False))
    in_news = None
    if apply_news_filter and windows_df is not None and not windows_df.empty and ts_col in df_out.columns:
        try:
            in_news = is_in_any_window(df_out[ts_col], windows_df[["start","end"]])
            df_out['in_news'] = in_news
        except Exception:
            in_news = None
    # Guard
    tg = st.session_state.get('trade_guard')
    guard_ok = True
    try:
        if tg is not None:
            s = tg.state()
            guard_ok = not bool(s.get('in_cooldown', False))
    except Exception:
        guard_ok = True
    # ドリフト自動停止
    drift_block = bool(st.session_state.get('auto_pause_on_drift', True) and (st.session_state.get('drift_state') == 'alert'))
    # 運用モード
    enable = bool(st.session_state.get('enable_trading', False))
    # シグナル列
    sig = df_out[signal_col] if signal_col in df_out.columns else True
    # ハードニュースのブロック条件
    news_block = False
    if apply_news_filter and isinstance(in_news, (pd.Series, np.ndarray, list)):
        news_block = in_news
    elif apply_news_filter:
        news_block = False
    # 最終ゲート
    gate = enable and guard_ok and (not drift_block)
    df_out['trade_ok'] = gate & (sig.astype(bool)) & (~news_block if isinstance(news_block, (pd.Series, np.ndarray, list)) else (not news_block))
    # ゲート統計を保存（サマリで表示）
    try:
        total = int(len(df_out))
        ok_cnt = int(df_out['trade_ok'].sum())
        pass_rate = (ok_cnt / total) if total > 0 else None
        st.session_state['gate_stats'] = {'total': total, 'trade_ok_count': ok_cnt, 'pass_rate': pass_rate}
        # 履歴保存（スパークライン用）
        st.session_state.setdefault('gate_pass_hist', [])
        pr_val = float(pass_rate) if isinstance(pass_rate, (int, float)) and np.isfinite(pass_rate) else np.nan
        st.session_state['gate_pass_hist'] = (st.session_state['gate_pass_hist'] + [pr_val])[-200:]
    except Exception:
        pass
    if add_columns:
        df_out['gate_enable'] = enable
        df_out['gate_guard_ok'] = guard_ok
        df_out['gate_drift_block'] = drift_block
        if apply_news_filter:
            df_out['gate_news_block'] = news_block if isinstance(news_block, (pd.Series, np.ndarray, list)) else bool(news_block)
    return df_out

def prepare_df_feats_for_inference(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    raw_df: 列に timestamp, open, high, low, close, volume を含む DataFrame（昇順）
    戻り値: df_feats（学習と同じ特徴量列 + timestamp）。欠損は0で埋め。
    """
    if not {"timestamp","open","high","low","close"}.issubset(set(c.lower() for c in raw_df.columns)):
        # 大文字ケースから標準化
        rename_map = {}
        for c in raw_df.columns:
            lc = c.lower()
            if lc in ["timestamp","open","high","low","close","volume"]:
                rename_map[c] = lc
        raw_df = raw_df.rename(columns=rename_map)

    # 時刻の整形と昇順保証
    raw_df = raw_df.copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df = raw_df.sort_values("timestamp").reset_index(drop=True)
    assert raw_df["timestamp"].is_monotonic_increasing

    # 1) 基本特徴
    base_feats = build_features(raw_df)
    # 2) 追加特徴（リークなし）
    raw_l = raw_df.rename(columns=str.lower)
    df_feats = augment_features(base_feats, raw_l)

    # 推論で使うので欠損は0埋め（学習側と同じ方針）
    df_feats = df_feats.fillna(0.0)

    # 重要：timestamp は残す（セッション判定に使う）
    if "timestamp" not in df_feats.columns:
        df_feats.insert(0, "timestamp", raw_df["timestamp"].values)

    return df_feats

@st.cache_resource
def _load_model_and_meta():
    """モデルとメタ情報を一度だけ読み込むキャッシュ関数。
    （重複定義されていたため統合）"""
    model, use_cols = load_break_model("models/break_model.joblib")
    meta = load_break_meta("models/break_meta.json")
    return model, use_cols, meta

# --- 復元用ダミークラス（呼び出さない）---

import numpy as np
from typing import Iterator, Tuple
import json, datetime as dt
import pandas as pd

def _pick_session_from_ts(ts: pd.Timestamp):
    h = ts.hour
    if 9 <= h < 15:  return "Tokyo"
    if 16<= h < 24:  return "London"
    if h>=22 or h<5: return "NY"
    return None

def pick_theta_for_now(meta):
    # 時間帯ごとθ → なければグローバルθでフォールバック
    now = pd.Timestamp.now(tz="Asia/Tokyo")
    sess = _pick_session_from_ts(now)
    tbs  = meta.get("theta_by_session", {})
    if sess and sess in tbs and "theta" in tbs[sess]:
        return float(tbs[sess]["theta"])
    return float(meta.get("threshold", 0.93))

# ---------------- Adaptive θ 基礎実装 ----------------
def compute_adaptive_theta(base_theta: float,
                           atr_short: float,
                           atr_long: float,
                           coverage_recent: float,
                           target_coverage: float,
                           k_atr: float,
                           k_cov: float,
                           theta_min: float,
                           theta_max: float) -> float:
    """ATR 比率と coverage 偏差で θ を微調整。
    θ' = base * (1 + k_atr * (atr_ratio-1)) + k_cov * (coverage_recent - target)
    クリップ後返す。
    """
    if not atr_long or atr_long <= 0 or not atr_short or atr_short <= 0:
        atr_ratio = 1.0
    else:
        atr_ratio = float(atr_short / atr_long)
    theta_prime = base_theta * (1.0 + k_atr * (atr_ratio - 1.0)) + k_cov * (coverage_recent - target_coverage)
    theta_prime = max(theta_min, min(theta_max, theta_prime))
    st.session_state.setdefault('theta_meta', {})['last'] = dict(
        base=base_theta,
        atr_short=atr_short,
        atr_long=atr_long,
        atr_ratio=atr_ratio,
        coverage_recent=coverage_recent,
        target_coverage=target_coverage,
        k_atr=k_atr,
        k_cov=k_cov,
        theta_adaptive=theta_prime,
        theta_min=theta_min,
        theta_max=theta_max,
    )
    return theta_prime

def _rolling_atr(series: pd.Series, period: int) -> float:
    if series is None or len(series) < period + 1:
        return float('nan')
    tr = series.diff().abs()
    if len(tr) < period:
        return float('nan')
    return float(tr.rolling(period).mean().iloc[-1])

def update_adaptive_theta(prob_history: list, window: int = 200) -> float:
    """prob_history: list[(prob, timestamp)] 直近 window を用いて coverage を求め θ 更新"""
    if not st.session_state.get('use_adaptive_theta', False):
        return st.session_state.get('theta_base', 0.60)
    base_theta = st.session_state.get('theta_base', 0.60)
    theta_min = st.session_state.get('theta_min', 0.40)
    theta_max = st.session_state.get('theta_max', 0.85)
    target_cov = st.session_state.get('theta_target_cov', 0.08)
    k_atr = st.session_state.get('k_atr', 0.50)
    k_cov = st.session_state.get('k_cov', 0.50)
    atr_short_p = st.session_state.get('atr_short_period', 14)
    atr_long_p  = st.session_state.get('atr_long_period', 56)

    closes = st.session_state.get('recent_closes', [])
    close_ser = pd.Series(closes) if closes else pd.Series(dtype=float)
    atr_short = _rolling_atr(close_ser, atr_short_p)
    atr_long  = _rolling_atr(close_ser, atr_long_p)

    sub = prob_history[-window:] if len(prob_history) > window else prob_history
    coverage_recent = 0.0
    if sub:
        coverage_recent = sum(1 for p,_ in sub if p >= base_theta) / len(sub)
    adaptive_theta = compute_adaptive_theta(base_theta, atr_short, atr_long,
                                            coverage_recent, target_cov,
                                            k_atr, k_cov, theta_min, theta_max)
    # ドリフトアラート中は θ にブーストを加える（厳しめに）
    drift_bump = 0.0
    if st.session_state.get('theta_drift_bump_active', False):
        drift_bump = float(st.session_state.get('theta_bump_drift', 0.03))
    theta_final = max(theta_min, min(theta_max, adaptive_theta + drift_bump))
    st.session_state['theta_adaptive'] = theta_final
    return theta_final

# ---------------- Drift 総合指標（PSI/KL/JS/H） ----------------
def compute_drift_score(dm: dict,
                        w: dict | None = None,
                        caps: dict | None = None) -> float:
    """dm: {'psi','kl','js','hellinger'}
    各指標を0-1に正規化（capで頭打ち）後、重み付き合算して0-1の総合スコアを返す。
    既定重み: psi 0.5, kl 0.2, js 0.2, h 0.1
    既定cap: psi 0.5, kl 0.5, js 0.5, h 1.0
    """
    w = w or st.session_state.get('drift_weights', {'psi':0.5,'kl':0.2,'js':0.2,'h':0.1})
    caps = caps or st.session_state.get('drift_caps', {'psi':0.5,'kl':0.5,'js':0.5,'h':1.0})
    psi = float(dm.get('psi', float('nan')))
    kl  = float(dm.get('kl',  float('nan')))
    js  = float(dm.get('js',  float('nan')))
    h   = float(dm.get('hellinger', float('nan')))
    def nz(x):
        return x if np.isfinite(x) and x>=0 else 0.0
    psi_n = min(1.0, nz(psi) / max(1e-9, caps.get('psi',0.5)))
    kl_n  = min(1.0, nz(kl)  / max(1e-9, caps.get('kl',0.5)))
    js_n  = min(1.0, nz(js)  / max(1e-9, caps.get('js',0.5)))
    h_n   = min(1.0, nz(h)   / max(1e-9, caps.get('h', 1.0)))
    score = (psi_n * w.get('psi',0.5) +
             kl_n  * w.get('kl', 0.2) +
             js_n  * w.get('js', 0.2) +
             h_n   * w.get('h',  0.1))
    # 重み和で正規化（安全）
    wsum = sum([w.get('psi',0.5), w.get('kl',0.2), w.get('js',0.2), w.get('h',0.1)])
    if wsum <= 0: wsum = 1.0
    return float(np.clip(score / wsum, 0.0, 1.0))

def _now_session_name(ts: pd.Timestamp) -> str:
    try:
        return in_sessions(ts)
    except Exception:
        return "Other"

def is_in_news_window(ts: pd.Timestamp, windows_df: pd.DataFrame, *, news_win_minutes: int = 30,
                      imp_min: int = 3, mode_label: str = "重要度別（赤影と同じ）") -> bool:
    if mode_label == "重要度別（赤影と同じ）":
        return is_suppressed(ts, windows_df) if windows_df is not None else False
    else:
        # 一律±分（過去イベントのみ考慮、ここでは windows_df を使わず news_df を参照しない設計に合わせる）
        # 既存コードの _is_suppressed_at の簡易版
        return False

def compute_final_theta_for_time(ts: pd.Timestamp, meta: dict, windows_df: pd.DataFrame) -> tuple[float, dict]:
    """最終θを一本化して返す。
    優先順位: base→session補正（max）→ドリフト→ニュース（ソフト時のみ）→クリップ
    breakdown で各寄与を返す。
    """
    theta_min = st.session_state.get('theta_min', 0.40)
    theta_max = st.session_state.get('theta_max', 0.85)
    base_theta = st.session_state.get('theta_adaptive', st.session_state.get('theta_base', 0.60))

    # セッション別（meta优先）。競合を避けるため base と session_theta の“高い方”を採用
    try:
        session_theta = pick_theta_for_now(meta)
    except Exception:
        session_theta = base_theta
    base_after_session = max(float(base_theta), float(session_theta))
    b_session = base_after_session - base_theta

    # ドリフトブースト
    b_drift = float(st.session_state.get('theta_bump_drift', 0.03)) if st.session_state.get('theta_drift_bump_active', False) else 0.0

    # ニュース（ソフト抑制時のみ）
    in_news = is_in_news_window(ts, windows_df, news_win_minutes=st.session_state.get('news_win', 30),
                                imp_min=st.session_state.get('news_imp_min', 3), mode_label=st.session_state.get('news_filter_mode', "重要度別（赤影と同じ）"))
    st.session_state['soft_suppress_active'] = bool(in_news and st.session_state.get('use_soft_suppress', False))
    b_news = float(st.session_state.get('theta_bump_in_news', 0.03)) if st.session_state['soft_suppress_active'] else 0.0

    theta_final = base_after_session + b_drift + b_news
    theta_final = max(theta_min, min(theta_max, theta_final))
    breakdown = {
        'base': float(base_theta),
        'session_used': float(session_theta),
        'session_bump': float(b_session),
        'drift_bump': float(b_drift),
        'news_bump': float(b_news),
        'min': float(theta_min),
        'max': float(theta_max),
        'in_news': bool(in_news),
    }
    return float(theta_final), breakdown

def render_meta_summary(df: pd.DataFrame, windows_df: pd.DataFrame, meta: dict):
    # 最新時刻
    last_ts = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) and len(df.index)>0 else pd.Timestamp.now(tz=JST)
    # 最終θ
    theta_final, br = compute_final_theta_for_time(last_ts, meta, windows_df)
    st.session_state['theta_final'] = theta_final
    st.session_state['theta_breakdown'] = br
    # EV per trade
    try:
        ev_meta = load_break_meta("models/break_meta.json")
        evpt = float(ev_meta.get("ev_per_trade", float('nan')))
    except Exception:
        evpt = float('nan')
    # Drift
    dm = st.session_state.get('drift_meta', {})
    score = compute_drift_score(dm)
    # Guard
    tg = st.session_state.get('trade_guard')
    g_state = tg.state() if tg else {}
    # News
    ns = st.session_state.get('news_stats', {'hits':0,'miss':0,'size':0})
    in_news = br.get('in_news', False)
    # Pending
    pend = st.session_state.get('pending_trades')
    pend_cnt = len(pend) if isinstance(pend, (list, tuple)) else 0

    # --- 履歴を更新（スパークライン用）---
    st.session_state.setdefault('theta_hist', [])
    st.session_state.setdefault('drift_score_hist', [])
    st.session_state.setdefault('evpt_hist', [])
    st.session_state['theta_hist'] = (st.session_state['theta_hist'] + [float(theta_final)])[-200:]
    st.session_state['drift_score_hist'] = (st.session_state['drift_score_hist'] + [float(score)])[-200:]
    st.session_state['evpt_hist'] = (st.session_state['evpt_hist'] + [float(evpt) if np.isfinite(evpt) else np.nan])[-200:]

    st.markdown("### 運用サマリ")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("EV/trade", f"{evpt:.4f}" if np.isfinite(evpt) else "N/A")
        st.caption("モデルメタより")
    with c2:
        st.metric("θ 最終", f"{theta_final:.3f}")
        st.caption(f"base {br['base']:.3f} / sess {br['session_used']:.3f} / drift +{br['drift_bump']:.2f} / news +{br['news_bump']:.2f}")
    with c3:
        st.metric("ニュース", "抑制中" if in_news else "通常")
        st.caption(f"LRU {ns.get('size',0)} / H {ns.get('hits',0)} / M {ns.get('miss',0)}")
    with c4:
        st.metric("ドリフト", f"{int(round(score*100))}/100")
        st.caption(f"PSI {dm.get('psi', float('nan')):.3f} / {dm.get('severity','n/a')}")
    with c5:
        if g_state:
            cd = "ON" if g_state.get('in_cooldown') else "OFF"
            st.metric("Guard", f"{cd}")
            st.caption(f"day {g_state.get('day_trades','?')}/{g_state.get('max_day_trades','?')}")
        else:
            st.metric("Guard", "n/a")
            st.caption("未初期化")
    with c6:
        st.metric("Pending", f"{pend_cnt}")
        st.caption("未確定建玉数（存在すれば）")

    # --- スパークライン（薄い小型チャート） ---
    try:
        import plotly.graph_objects as go
        def spark(y, color="#8ecae6"):
            fig = go.Figure(go.Scatter(y=y, mode="lines", line=dict(color=color, width=1)))
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=60, xaxis=dict(visible=False), yaxis=dict(visible=False))
            return fig
        srow = st.columns(4)
        with srow[0]:
            st.plotly_chart(spark(st.session_state['evpt_hist'], "#90be6d"), use_container_width=True)
            st.caption("EV spark")
        with srow[1]:
            st.plotly_chart(spark(st.session_state['theta_hist'], "#577590"), use_container_width=True)
            st.caption("θ spark")
        with srow[2]:
            st.plotly_chart(spark(st.session_state['drift_score_hist'], "#f8961e"), use_container_width=True)
            st.caption("Drift spark")
        with srow[3]:
            st.session_state.setdefault('gate_pass_hist', [])
            spark_vals = [v*100.0 if isinstance(v, (int,float)) and np.isfinite(v) else np.nan for v in st.session_state['gate_pass_hist']]
            st.plotly_chart(spark(spark_vals, "#118ab2"), use_container_width=True)
            st.caption("Gate % spark")
    except Exception:
        pass

    # --- ゲート適用の件数とパス率 ---
    gate_stats = st.session_state.get('gate_stats')
    if isinstance(gate_stats, dict):
        gcols = st.columns(2)
        with gcols[0]:
            st.metric("trade_ok件数(直近)", f"{gate_stats.get('trade_ok_count', 0)}")
        with gcols[1]:
            pr = gate_stats.get('pass_rate')
            st.metric("ゲート通過率", f"{pr*100:.1f}%" if isinstance(pr, (int,float)) else "N/A")

class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, test_size: int | None = None, embargo: int = 0):
        self.n_splits = int(n_splits)
        self.test_size = None if test_size is None else int(test_size)
        self.embargo = max(0, int(embargo))
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        # アプリでは学習分割は使いません
        raise RuntimeError("PurgedTimeSeriesSplit is a pickle shim for unpickling only.")

# --- モデル読み込み ---

# （削除）ここにあった _load_model_and_meta の重複定義を統合済み。

# 推論時の利用例
model, use_cols, meta = _load_model_and_meta()
# raw_df = ...（既存のデータ取得処理）
# df_feats = prepare_df_feats_for_inference(raw_df)
# pred = predict_with_session_theta(df_feats, model, use_cols, meta)
import requests
import pandas as pd
# ======== ローリング（時点ごと再計算）バックテスト ========
def _build_windows_until(t_end: pd.Timestamp, imp_threshold: int) -> pd.DataFrame:
    """
    t_end（含む）までのニュースのみから、重要度別ウィンドウを構築
    """
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=["start","end","importance","title"])
    past_events = news_df[news_df["time"] <= t_end]
    if past_events.empty:
        return pd.DataFrame(columns=["start","end","importance","title"])
    return build_event_windows(past_events, imp_threshold=imp_threshold, mapping=imp_map)

# --- ニュース抑制 LRU キャッシュ ---
from collections import OrderedDict
_NEWS_CACHE_MAX = 300
def _get_news_windows_until_cached(t_end: pd.Timestamp, imp_threshold: int) -> pd.DataFrame:
    key = (pd.Timestamp(t_end).floor('T'), int(imp_threshold))
    cache = st.session_state.setdefault('news_win_cache', OrderedDict())
    stats = st.session_state.setdefault('news_stats', {'hits':0, 'miss':0, 'size':0})
    if key in cache:
        # move to end (recent)
        cache.move_to_end(key)
        stats['hits'] += 1
        return cache[key]
    dfw = _build_windows_until(t_end, imp_threshold)
    cache[key] = dfw
    cache.move_to_end(key)
    if len(cache) > _NEWS_CACHE_MAX:
        cache.popitem(last=False)
    stats['miss'] += 1
    stats['size'] = len(cache)
    return dfw
    return dfw

def _is_suppressed_at(ts: pd.Timestamp, win_df: pd.DataFrame, news_win_minutes: int,
                      imp_min: int, mode_label: str) -> bool:
    """
    フィルタ方式に応じて抑制判定（すべて過去限定）
    """
    if news_filter_mode == "重要度別（赤影と同じ）":
        if win_df is None or win_df.empty:
            return False
        return is_suppressed(ts, win_df)
    else:
        if news_df is None or news_df.empty:
            return False
        win = pd.Timedelta(minutes=news_win_minutes)
        cond = (news_df["time"] <= ts) & (news_df["importance"] >= imp_min) & news_df["time"].between(ts - win, ts + win)
        return bool(cond.any())

def backtest_rolling(df: pd.DataFrame,
                     fwd_n: int,
                     break_buffer_arg: float,
                     spread_pips: float,
                     news_win: int,
                     news_imp_min: int,
                     apply_news: bool,
                     signal_mode: str,
                     retest_wait_k_arg: int,
                     touch_buffer: float
                     ) -> pd.DataFrame:
    """
    ★ 未来情報リーケージを除去したローリング版バックテスト
      - 各時点 i で、過去のデータのみから：
          * ピボット→水平線（DBSCAN）
          * 回帰トレンド/チャネル
          * ニュース抑制ウィンドウ
        を再計算して判定します。
    """
    rows = []
    if len(df) <= fwd_n + 2:
        return pd.DataFrame(columns=["time","mode","level_or_val","dir","entry","exit","ret_pips","retest_index","retest_hit"])

    pv_local = pip_value("USDJPY")
    close_s = df["close"]

    # i の開始位置（最低限、ピボット/DBSCAN/回帰が安定する分だけ進める）
    min_start = max(2, reg_lookback, look * 4)

    for i in range(min_start, len(df) - fwd_n):
        # ---- 過去のみ抽出
        past = df.iloc[:i+1]
        t = past.index[-1]
        c  = float(past["close"].iloc[-1])
        l1 = float(past["low"].iloc[-2]); h1 = float(past["high"].iloc[-2])

        # ---- ニュース抑制（過去のみ）
        win_df_past = _get_news_windows_until_cached(t, news_imp_min) if apply_news else pd.DataFrame()

        # 抑制チェック（判定の“基準時刻”で見る）
        if apply_news and _is_suppressed_at(t, win_df_past, news_win, news_imp_min, signal_mode):
            continue

        # ---- 過去のみでレベル・トレンド再計算
        try:
            piv_hi_past, piv_lo_past = swing_pivots(past, look)
            lvls_past = horizontal_levels(piv_hi_past, piv_lo_past, eps=eps, min_samples=min_samples)
        except Exception:
            lvls_past = []

        tr_past = regression_trend(past, reg_lookback, use="low")

        # ---- モード別判定
        if signal_mode == "水平線ブレイク(終値)":
            for lv in lvls_past:
                if (c > lv + break_buffer_arg) and (l1 <= lv):
                    entry = c; exitp = float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, lv, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="水平ブレイク上", level_or_val=float(lv),
                                     dir="long", entry=entry, exit=exitp,
                                     ret_pips=(exitp-entry)/pv_local - spread_pips,
                                     retest_index=ri, retest_hit=rh))
                if (c < lv - break_buffer_arg) and (h1 >= lv):
                    entry = c; exitp = float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, lv, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="水平ブレイク下", level_or_val=float(lv),
                                     dir="short", entry=entry, exit=exitp,
                                     ret_pips=(entry-exitp)/pv_local - spread_pips,
                                     retest_index=ri, retest_hit=rh))

        elif signal_mode == "トレンドラインブレイク(終値)":
            if tr_past:
                tl = tr_past["y1"]
                if (c > tl + break_buffer_arg) and (l1 <= tl):
                    entry = c; exitp = float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, tl, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="TLブレイク上", level_or_val=float(tl),
                                     dir="long", entry=entry, exit=exitp,
                                     ret_pips=(exitp-entry)/pv_local - spread_pips,
                                     retest_index=ri, retest_hit=rh))
                if (c < tl - break_buffer_arg) and (h1 >= tl):
                    entry = c; exitp = float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, tl, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="TLブレイク下", level_or_val=float(tl),
                                     dir="short", entry=entry, exit=exitp,
                                     ret_pips=(entry-exitp)/pv_local - spread_pips,
                                     retest_index=ri, retest_hit=rh))

        elif signal_mode == "チャネル上抜け/下抜け(終値)":
            if tr_past and tr_past["sigma"] > 0:
                up = tr_past["y1"] + chan_k * tr_past["sigma"]
                dn = tr_past["y1"] - chan_k * tr_past["sigma"]
                if c > up + break_buffer_arg:
                    entry = c; exitp = float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, up, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="チャネル上抜け", level_or_val=float(up),
                                     dir="long", entry=entry, exit=exitp,
                                     ret_pips=(exitp-entry)/pv_local - spread_pips,
                                     retest_index=ri, retest_hit=rh))
                if c < dn - break_buffer_arg:
                    entry = c; exitp = float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, dn, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="チャネル下抜け", level_or_val=float(dn),
                                     dir="short", entry=entry, exit=exitp,
                                     ret_pips=(entry-exitp)/pv_local - spread_pips,
                                     retest_index=ri, retest_hit=rh))

        elif signal_mode == "リテスト指値(水平線)":
            K = int(retest_wait_k_arg)
            for lv in lvls_past:
                up_break = (c > lv + break_buffer_arg) and (l1 <= lv)
                dn_break = (c < lv - break_buffer_arg) and (h1 >= lv)

                # 上方向ブレイク後、K 本以内にリテスト→その“リテスト時刻”で約定
                if up_break:
                    for j in range(i+1, min(i+K, len(df)-fwd_n)):
                        t_j = df.index[j]
                        # リテスト時刻でもニュース抑制（過去ニュースのみ）
                        if apply_news:
                            win_df_j = _get_news_windows_until_cached(t_j, news_imp_min)
                            if _is_suppressed_at(t_j, win_df_j, news_win, news_imp_min, signal_mode):
                                continue
                        if abs(float(df["close"].iloc[j]) - lv) <= touch_buffer:
                            entry = float(df["close"].iloc[j]); exitp = float(df["close"].iloc[j+fwd_n])
                            ri, rh = compute_retest(close_s, lv, i, K, float(touch_buffer))
                            rows.append(dict(time=t_j, mode="リテスト(L)", level_or_val=float(lv),
                                             dir="long", entry=entry, exit=exitp,
                                             ret_pips=(exitp-entry)/pv_local - spread_pips,
                                             retest_index=ri, retest_hit=rh))
                            break

                # 下方向ブレイク後のリテスト
                if dn_break:
                    for j in range(i+1, min(i+K, len(df)-fwd_n)):
                        t_j = df.index[j]
                        if apply_news:
                            win_df_j = _get_news_windows_until_cached(t_j, news_imp_min)
                            if _is_suppressed_at(t_j, win_df_j, news_win, news_imp_min, signal_mode):
                                continue
                        if abs(float(df["close"].iloc[j]) - lv) <= touch_buffer:
                            entry = float(df["close"].iloc[j]); exitp = float(df["close"].iloc[j+fwd_n])
                            ri, rh = compute_retest(close_s, lv, i, K, float(touch_buffer))
                            rows.append(dict(time=t_j, mode="リテスト(S)", level_or_val=float(lv),
                                             dir="short", entry=entry, exit=exitp,
                                             ret_pips=(entry-exitp)/pv_local - spread_pips,
                                             retest_index=ri, retest_hit=rh))
                            break

    return pd.DataFrame(rows)
# -*- coding: utf-8 -*-
# Streamlit FX Auto Lines - 完全版 + News Shading + Flag/Pennant + H&S + Ghost Projection
# 黒背景・重要度別ニュースウィンドウ赤影・ソフト抑制・自動ライン/パターン/EV/ブレイク確率・手動再学習

import os, math, json, subprocess, sys, pathlib, re, warnings
from datetime import timedelta
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

def load_yf(symbol="JPY=X", period="60d", interval="15m"):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df = df.reset_index().rename(columns={"Datetime":"timestamp"})  # ダウンロード結果のindex→列へ
    return df

raw_df = load_yf("JPY=X", "60d", "15m")
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from dataclasses import dataclass
import pytz, joblib
from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings("ignore")

# --- 検証レポート表示関数 ---
def show_calibration_report():
    st.header("🔧 検証レポート：Calibration")
    path = "reports/break_calibration.json"
    png  = "reports/break_calibration.png"
    if not os.path.exists(path):
        st.info("レポートがありません。学習スクリプト実行後に自動生成されます。")
        return
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    cal = payload["calibration"] if "calibration" in payload else payload
    meta= payload.get("meta", {})
    st.write(f"**Brier**: `{cal['brier']:.6f}`  /  **ECE**: `{cal['ece']:.6f}`")
    st.write("**Meta**:", meta)
    # 表
    df = pd.DataFrame({
        "bin_left": cal["bin_edges"][:-1],
        "bin_right": cal["bin_edges"][1:],
        "prob_mean": cal["prob_mean"],
        "frac_pos": cal["frac_pos"],
        "count": cal["counts"],
        # NEW: CI 列（後方互換のため get で取得）
        "frac_lo": cal.get("frac_lo", [None]* (len(cal["bin_edges"])-1)),
        "frac_hi": cal.get("frac_hi", [None]* (len(cal["bin_edges"])-1)),
    }).astype({"count": int}, errors="ignore")
    st.dataframe(df, use_container_width=True)
    # 画像
    if os.path.exists(png):
        st.image(png, caption="Reliability Curve", use_container_width=True)


JST = pytz.timezone("Asia/Tokyo")

# --- メインタブ構成 ---
tabs = st.tabs(["トレード", "検証レポート"])
with tabs[0]:
    # ...既存のトレードUIコード...
    pass  # 既存のトレードUIはここに展開されているはず
with tabs[1]:
    show_calibration_report()

# ---------------- ダーク配色 ----------------
COLOR_BG = "#0b0f14"
COLOR_GRID = "#263238"
COLOR_TEXT = "#e0f2f1"
COLOR_LEVEL = "#00e5ff"            # 水平線
COLOR_TREND = "#ff9800"            # トレンドライン（オレンジ）
COLOR_CH_UP = "#e53935"            # チャネル上（赤）
COLOR_CH_DN = "#1e88e5"            # チャネル下（青）
# --- パターンごと色 ---
COLOR_TRIANGLE = "#8e24aa"          # トライアングル（紫）
COLOR_RECTANGLE = "#43a047"         # レクタングル（緑）
COLOR_DOUBLE_TOP = "#d81b60"        # ダブルトップ（ピンク）
COLOR_DOUBLE_BOTTOM = "#1976d2"     # ダブルボトム（青）
COLOR_FLAG = "#fbc02d"              # フラッグ/ペナント（黄）
COLOR_HS = "#6d4c41"                # ヘッド＆ショルダーズ（茶）
COLOR_CANDLE_UP_BODY = "#26a69a"
COLOR_CANDLE_UP_EDGE = "#66fff9"
COLOR_CANDLE_DN_BODY = "#ef5350"
COLOR_CANDLE_DN_EDGE = "#ff8a80"

# ---------------- Intraday制約クランプ ----------------
INTRADAY_SET = {"1m","2m","5m","15m","30m","60m","90m"}
MAX_PERIOD_BY_INTERVAL = {
    "1m":"7d","2m":"60d","5m":"60d","15m":"60d","30m":"60d",
    "60m":"730d","90m":"730d"
}
def clamp_period_for_interval(period: str, interval: str) -> str:
    if interval not in INTRADAY_SET: return period
    maxp = MAX_PERIOD_BY_INTERVAL.get(interval)
    if not maxp: return period
    def to_days(p: str) -> int:
        p = p.strip().lower()
        if p.endswith("d"):  return int(p[:-1])
        if p.endswith("mo"): return int(p[:-2]) * 30
        if p.endswith("y"):  return int(p[:-1]) * 365
        return 999999
    return maxp if to_days(period) > to_days(maxp) else period

# ======== APIキー読込（.env / secrets 両対応）========
@st.cache_resource(show_spinner=False)
def _load_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OpenAI API Key が見つかりません。`.env` を設定してください。")
        st.stop()

    # モデル名は.envで上書き可能。未設定なら安全デフォルト。
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    client = OpenAI(api_key=api_key)
    return client, model

client, DEFAULT_OAI_MODEL = _load_openai_client()

# ======== 状態収集（あなたの既存変数名に合わせて適宜修正）========
def collect_state_for_ai():
    return {
        "ticker": st.session_state.get("ticker", "JPY=X"),
        "period": st.session_state.get("period", "60d"),
        "interval": st.session_state.get("interval", "15m"),
        "H": st.session_state.get("H_pred", 12),
        "theta": st.session_state.get("theta", 0.60),
        "break_buffer": st.session_state.get("break_buf", 0.05),
        "K_retest": st.session_state.get("K", 10),
        "news": {
            "importance_threshold": st.session_state.get("imp_th", 3),
            "suppress_window_min": st.session_state.get("suppress_win", 30),
            "upcoming": st.session_state.get("upcoming_events", []),
        },
        "topk_levels": st.session_state.get("topk_levels", []),
    }

# ======== 返答抽出の安全版ユーティリティ ========
def _extract_text_from_responses(resp) -> str:
    """OpenAI Responses API / Chat Completions などの返却を安全に文字列化"""
    if resp is None:
        return ""

    # 新SDK（Responses API）: まず output_text を優先
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    # 新SDKで output の中に content が分割される形
    out = getattr(resp, "output", None)
    if out:
        chunks = []
        for item in out:
            content = getattr(item, "content", None) or []
            for part in content:
                # dict でも pydantic obj でも拾う
                if isinstance(part, dict):
                    if part.get("type") in ("output_text", "text") and "text" in part:
                        val = part["text"]
                        if isinstance(val, str):
                            chunks.append(val)
                else:
                    ptype = getattr(part, "type", None)
                    if ptype in ("output_text", "text"):
                        t = getattr(part, "text", None)
                        if isinstance(t, str):
                            chunks.append(t)
                        else:
                            # 一部で text.value に入るケース
                            v = getattr(t, "value", None)
                            if isinstance(v, str):
                                chunks.append(v)
        if chunks:
            return "\n".join(c for c in chunks if c).strip()

    # Chat Completions 互換
    try:
        v = resp.choices[0].message.content
        if isinstance(v, str) and v.strip():
            return v.strip()
    except Exception:
        pass

    # Completions 互換
    try:
        v = resp.choices[0].text
        if isinstance(v, str) and v.strip():
            return v.strip()
    except Exception:
        pass

    # dict 化されている場合の保険
    try:
        d = resp if isinstance(resp, dict) else resp.model_dump()
        for k in ("output_text", "output", "text", "content"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass

    return ""

def _fallback_answer(user_q: str) -> str:
    return (
        "（自動応答）うまく回答を生成できませんでした。少し具体化して再度お試しください。\n"
        "・質問を短く要点ごとに分ける\n"
        "・ネット/APIキー/レート制限の状態を確認\n"
    )

# ======== コパイロット呼び出し（例外をUIに出す）========
def ask_copilot(app_state: dict, user_question: str) -> str:
    system = (
        "You are an FX breakout trading copilot specialized in USDJPY 15m.\n"
        "Use the JSON app state to give short, practical guidance.\n"
        "Output format:\n"
        "- One-line conclusion\n"
        "- 3 bullet reasons (concise)\n"
        "- Next action (one line)\n"
        "- Disclaimer (one line, no guarantees)\n"
        "Never guarantee profits. Be cautious around news windows."
    )

    # --- 呼び出し & 抽出 ---
    raw = None
    ans = ""
    error_msg = None
    model_used = os.getenv("OPENAI_MODEL", DEFAULT_OAI_MODEL)

    try:
        raw = client.responses.create(
            model=model_used,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"APP_STATE_JSON:\n{json.dumps(app_state, ensure_ascii=False)}"},
                {"role": "user", "content": f"QUESTION:\n{user_question or '(空)'}"},
            ],
            max_output_tokens=2000,
        )
        ans = _extract_text_from_responses(raw).strip()
        # デバッグ用: レスポンス内容を表示（expanderのネスト回避）
        st.markdown("### APIレスポンス詳細")
        st.write(raw)
        if not ans:
            st.error("AI応答が空でした。APIレスポンス内容を確認してください。")
            st.write("APIレスポンス:", raw)
            raise RuntimeError("AI応答が空でした（output_text / output / choices からテキスト取得不可）。")
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        ans = _fallback_answer(user_question)

    # --- デバッグ情報（expanderを使わず container で）---
    debug_dict = {
        "model": model_used,
        "question": user_question,
        "app_state_keys": list(app_state.keys()) if isinstance(app_state, dict) else None,
        "extracted_len": len(ans) if isinstance(ans, str) else None,
        "had_error": error_msg is not None,
        "error": error_msg,
    }
    with st.container():
        st.markdown("**デバッグ情報**")
        st.write(debug_dict)

    return ans

# ======== サイドバー：入力だけ。表示はメインに回す ========
with st.sidebar.expander("🤖 コパイロット（gpt-5-mini）", expanded=False):
    user_q = st.text_area("相談内容", height=90, placeholder="例）この設定でNY時間は θ を上げるべき？")
    if st.button("AIに相談"):
        if not (user_q or "").strip():
            st.warning("質問を入力してください。")
        else:
            with st.spinner("AIが分析中..."):
                app_state = collect_state_for_ai()
                ans = ask_copilot(app_state, user_q)
                st.session_state["copilot_answer"] = ans  # ← 状態に保存

# ======== メインエリアで“直近の回答”を表示 ========
if st.session_state.get("copilot_answer"):
    st.subheader("🤖 コパイロットの回答")
    st.write(st.session_state["copilot_answer"])

# ======== 免責は従来どおり（常時表示） ========
st.markdown("""
---
**免責事項**：本コパイロットの提案は教育・参考目的です。将来の利益を保証するものではありません。  
最終的な投資判断はご自身の責任でお願いします。
""")

# ---------------- Utils ----------------
def ensure_jst_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    return df.copy().set_index(idx.tz_convert(JST))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def in_sessions(ts: pd.Timestamp) -> str:
    h = ts.hour
    if 9 <= h < 15: return "Tokyo"
    if 16<= h < 24: return "London"
    if h>=22 or h<5: return "NY"
    return "Other"

def pip_value(pair="USDJPY"):
    return 0.01  # USDJPY想定

def _select_first(values) -> str:
    return list(dict.fromkeys(map(str, values)))[0]

def normalize_ohlcv(df: pd.DataFrame, symbol: str | None) -> pd.DataFrame:
    """yfinanceの単層/多層列を open/high/low/close に統一"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    if not isinstance(df.columns, pd.MultiIndex):
        df = df.rename(columns=lambda c: str(c).strip()).rename(columns=str.lower)
        need = {"open","high","low","close"}
        if not need.issubset(set(df.columns)):
            colmap = {}
            for want in ["Open","High","Low","Close","Adj Close","Volume"]:
                cand = [c for c in df.columns if c.lower().startswith(want.lower())]
                if cand: colmap[cand[0]] = want.lower()
            df = df.rename(columns=colmap)
            if not need.issubset(set(df.columns)):
                raise ValueError(f"必須列が見つかりません: {need - set(df.columns)}")
        return df

    # MultiIndex対応
    cols = df.columns
    lvl0, lvl1 = set(map(str, cols.get_level_values(0))), set(map(str, cols.get_level_values(1)))
    fields = {"Open","High","Low","Close","Adj Close","Volume"}
    if fields & lvl0 and not (fields & lvl1):
        df = df.swaplevel(0,1,axis=1)
        cols = df.columns

    if fields & set(map(str, cols.get_level_values(1))):
        tickers = list(dict.fromkeys(map(str, cols.get_level_values(0))))
        pick_ticker = symbol if (symbol and symbol in tickers) else _select_first(cols.get_level_values(0))
        sub = df.xs(pick_ticker, axis=1, level=0)
        if isinstance(sub.columns, pd.MultiIndex):
            try: sub = sub.droplevel(0, axis=1)
            except Exception: sub.columns = ["_".join(map(str,c)) for c in sub.columns]
        want = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in sub.columns]
        if not {"Open","High","Low","Close"}.issubset(set(want)):
            def _find(name):
                for c in sub.columns:
                    if str(c).lower()==name.lower(): return c
                for c in sub.columns:
                    if name.lower() in str(c).lower(): return c
                return None
            pick = {k:_find(k) for k in ["Open","High","Low","Close","Adj Close","Volume"]}
            if not all(pick[k] for k in ["Open","High","Low","Close"]):
                raise ValueError("OHLCV列の抽出に失敗（異例のMultiIndex）。")
            want = [pick[k] for k in ["Open","High","Low","Close"] if pick[k] is not None]
            if pick.get("Adj Close"): want.append(pick["Adj Close"])
            if pick.get("Volume"):    want.append(pick["Volume"])
        out = sub[want].copy()
        out.columns = [str(c).lower() for c in out.columns]
        return out

    df_flat = df.copy()
    df_flat.columns = ["_".join(map(str, c)) for c in df_flat.columns]
    mapping = {}
    for fld in ["Open","High","Low","Close","Adj Close","Volume"]:
        cand = [c for c in df_flat.columns if c.split("_")[-1].lower()==fld.lower() or fld.lower() in c.lower()]
        if cand: mapping[fld]=cand[0]
    if not {"Open","High","Low","Close"}.issubset(mapping.keys()):
        raise ValueError("OHLCV列の抽出に失敗（未知の列構成）。")
    cols_ordered = [mapping[k] for k in ["Open","High","Low","Close"] if k in mapping]
    if "Adj Close" in mapping: cols_ordered.append(mapping["Adj Close"])
    if "Volume" in mapping:    cols_ordered.append(mapping["Volume"])
    out = df_flat[cols_ordered].copy()
    out.columns = [c.split("_")[-1].lower() for c in out.columns]
    return out

# ---------------- サイドバー ----------------
st.sidebar.title("設定")
symbol = st.sidebar.text_input("ティッカー（USDJPY）", value="JPY=X", help="yfinanceでUSDJPYは 'JPY=X'")
period_raw = st.sidebar.selectbox("取得期間", ["7d","14d","30d","60d","90d","180d","1y"], index=2)
interval = st.sidebar.selectbox("足種", ["5m","15m","30m","60m","1d"], index=1)

# === 推奨行動（意思決定ポリシー）サイドバー ===
with st.sidebar.expander("🧭 推奨行動（意思決定ポリシー）", expanded=False):
    news_mode = st.selectbox("ニュース時の基本動作", ["hard","soft"], index=0,
                             help="hard=見送り固定 / soft=θを上げる")
    min_ev_r = st.number_input("EV/Rの下限", value=0.00, step=0.01, format="%.2f")
    spread_max = st.number_input("許容スプレッド（同単位）", value=0.03, step=0.01)
    wick_ratio_max = st.number_input("長ヒゲ閾値（ヒゲ/実体）", value=2.5, step=0.5)
    prefer_limit = st.toggle("リテスト指値を優先（“入らない勇気”）", value=True)
    bump_low  = st.number_input("θ補正: lowボラ +", value=0.00, step=0.01, format="%.2f")
    bump_mid  = st.number_input("θ補正: midボラ +", value=0.02, step=0.01, format="%.2f")
    bump_high = st.number_input("θ補正: highボラ +", value=0.03, step=0.01, format="%.2f")

params = DecisionParams(
    min_ev_r=min_ev_r,
    theta_bump_by_regime={"low":bump_low,"mid":bump_mid,"high":bump_high},
    theta_bump_in_news=0.03,
    news_mode=news_mode,
    spread_max=spread_max,
    wick_ratio_max=wick_ratio_max,
    prefer_limit_retest=prefer_limit,
)


# --- OANDA API設定 ---
st.sidebar.markdown("---")
with st.sidebar.expander("OANDA API設定", expanded=False):
    oanda_token = st.text_input("OANDA APIトークン", type="password", help="OANDAのAPIトークンを入力")
    oanda_account = st.text_input("OANDAアカウント番号", help="OANDAのアカウント番号を入力")
    oanda_env = st.selectbox("OANDA環境", ["practice", "live"], index=0, help="practice=デモ, live=本番")

st.sidebar.markdown("---")
st.sidebar.subheader("取引コスト設定（pips）")
fee_commission = st.sidebar.number_input("手数料（往復）", min_value=0.0, max_value=5.0, value=0.00, step=0.01)
fee_slippage  = st.sidebar.number_input("スリッページ（平均）", min_value=0.0, max_value=5.0, value=0.20, step=0.01)
fee_gap       = st.sidebar.number_input("ギャップ控除（期待値）", min_value=0.0, max_value=10.0, value=0.00, step=0.01)
extra_cost_pips = float(fee_commission + fee_slippage + fee_gap)

st.sidebar.subheader("自動更新")
auto_refresh = st.sidebar.checkbox("自動で再取得（ページ再読み込み）", value=True)
refresh_secs = st.sidebar.slider("更新間隔（秒）", 30, 600, 600, help="15分足は60〜180秒が目安")
try:
    from streamlit_autorefresh import st_autorefresh
    if auto_refresh:
        st_autorefresh(interval=refresh_secs * 1000, limit=None, key="fx_autorefresh")
except Exception:
    if auto_refresh:
        from streamlit.components.v1 import html
        html(f"""<script>setTimeout(function(){{window.location.reload();}}, {int(refresh_secs*1000)});</script>""", height=0)

st.sidebar.markdown("---")
st.sidebar.subheader("シグナル条件")
signal_mode = st.sidebar.selectbox(
    "種別を選択",
    ["水平線ブレイク(終値)", "トレンドラインブレイク(終値)", "チャネル上抜け/下抜け(終値)", "リテスト指値(水平線)"],
    index=0
)
retest_wait_k_base = st.sidebar.slider("リテスト待機本数K", 3, 30, 10)
st.sidebar.caption("ブレイク後、K本以内にライン/バンドへ戻ったかで『リテストあり/なし』を判定（指数0〜1も算出）")

with st.sidebar.expander("Adaptive θ", expanded=False):
    st.session_state.setdefault('theta_base', 0.60)
    st.session_state.setdefault('use_adaptive_theta', False)
    st.session_state['use_adaptive_theta'] = st.checkbox("有効化", value=st.session_state['use_adaptive_theta'])
    st.session_state['theta_base'] = st.number_input("基礎θ", 0.30, 0.99, float(st.session_state['theta_base']), 0.01)
    st.session_state['theta_min'] = st.number_input("θ最小", 0.10, 0.99, st.session_state.get('theta_min', 0.40), 0.01)
    st.session_state['theta_max'] = st.number_input("θ最大", 0.10, 0.99, st.session_state.get('theta_max', 0.85), 0.01)
    st.session_state['theta_target_cov'] = st.number_input("目標Coverage", 0.01, 0.50, st.session_state.get('theta_target_cov', 0.08), 0.01)
    st.session_state['k_atr'] = st.slider("ATR影響係数 k_atr", 0.0, 1.5, float(st.session_state.get('k_atr', 0.5)), 0.05)
    st.session_state['k_cov'] = st.slider("Coverage影響係数 k_cov", 0.0, 2.0, float(st.session_state.get('k_cov', 0.5)), 0.05)
    st.session_state['atr_short_period'] = st.slider("ATR短期", 5, 60, int(st.session_state.get('atr_short_period', 14)))
    st.session_state['atr_long_period']  = st.slider("ATR長期", 20, 200, int(st.session_state.get('atr_long_period', 56)))
    st.caption("ATR比率 + 発火率偏差で θ を微調整します")

# --- Adaptive θ プリセット（Conservative/Balanced/Aggressive）---
with st.sidebar.expander("Adaptive θ プリセット", expanded=False):
    c1, c2, c3 = st.columns(3)
    def _apply_preset(name: str):
        if name == "Conservative":
            st.session_state['theta_base'] = 0.64
            st.session_state['theta_min'] = 0.50
            st.session_state['theta_max'] = 0.90
            st.session_state['theta_target_cov'] = 0.06
            st.session_state['k_atr'] = 0.70
            st.session_state['k_cov'] = 0.30
        elif name == "Balanced":
            st.session_state['theta_base'] = 0.60
            st.session_state['theta_min'] = 0.45
            st.session_state['theta_max'] = 0.88
            st.session_state['theta_target_cov'] = 0.08
            st.session_state['k_atr'] = 0.50
            st.session_state['k_cov'] = 0.50
        else:  # Aggressive
            st.session_state['theta_base'] = 0.56
            st.session_state['theta_min'] = 0.40
            st.session_state['theta_max'] = 0.85
            st.session_state['theta_target_cov'] = 0.10
            st.session_state['k_atr'] = 0.30
            st.session_state['k_cov'] = 0.70
        st.toast(f"プリセット適用: {name}")
    with c1:
        if st.button("Conservative"):
            _apply_preset("Conservative")
    with c2:
        if st.button("Balanced"):
            _apply_preset("Balanced")
    with c3:
        if st.button("Aggressive"):
            _apply_preset("Aggressive")

# --- ドリフト監視（ヒステリシス＋自動ガード/θブースト）---
with st.sidebar.expander("📉 データドリフト監視", expanded=False):
    st.session_state.setdefault('drift_meta', {})
    # しきい値と作用
    hi_thr = st.number_input("PSI High（一時停止）", 0.0, 1.0, float(st.session_state.get('psi_hi', 0.25)), 0.01, key="psi_hi")
    lo_thr = st.number_input("PSI Low（復帰）", 0.0, 1.0, float(st.session_state.get('psi_lo', 0.15)), 0.01, key="psi_lo")
    theta_bump_drift = st.number_input("ドリフト時 θ 追加", 0.00, 0.20, float(st.session_state.get('theta_bump_drift', 0.03)), 0.01, key="theta_bump_drift")
    auto_pause = st.checkbox("自動で運用一時停止（EVゲートOFF）", value=st.session_state.get('auto_pause_on_drift', True), key="auto_pause_on_drift")
    cooldown_sec = st.number_input("通知クールダウン(秒)", 0, 3600, int(st.session_state.get('drift_alert_cooldown_sec', 120)), 10, key="drift_alert_cooldown_sec")
    # 総合スコアで自動停止するオプション
    use_comp = st.checkbox("総合スコアで自動停止を判定", value=st.session_state.get('drift_use_composite', True), key="drift_use_composite")
    score_hi = st.slider("スコア High（一時停止）", 0.0, 1.0, float(st.session_state.get('drift_score_hi', 0.55)), 0.01, key="drift_score_hi")
    score_lo = st.slider("スコア Low（復帰）", 0.0, 1.0, float(st.session_state.get('drift_score_lo', 0.40)), 0.01, key="drift_score_lo")
    # 現在値（UI下段）
    dm = st.session_state.get('drift_meta') or {}
    dscore = compute_drift_score(dm)
    st.caption(f"PSI={dm.get('psi', float('nan')):.3f} / KL={dm.get('kl', float('nan')):.3f} / JS={dm.get('js', float('nan')):.3f} / H={dm.get('hellinger', float('nan')):.3f} / Score={dscore:.3f}")
    # ヒステリシス状態を保持
    st.session_state.setdefault('drift_state', 'ok')
    state = st.session_state['drift_state']
    psi_now = float(dm.get('psi', float('nan')))
    # 判定は総合スコア/PSIのどちらか（オプション）
    if use_comp:
        now_ts = pd.Timestamp.utcnow()
        last_toast = st.session_state.get('drift_last_toast_ts')
        def _can_toast():
            if not isinstance(last_toast, pd.Timestamp):
                return True
            return (now_ts - last_toast).total_seconds() >= int(st.session_state.get('drift_alert_cooldown_sec', 120))
        if state in ('ok','recover') and dscore >= score_hi:
            st.session_state['drift_state'] = 'alert'
            if auto_pause:
                st.session_state['enable_trading'] = False
            if _can_toast():
                st.toast("Drift ScoreがHighを超過: alert に遷移", icon="⚠️")
                st.session_state['drift_last_toast_ts'] = now_ts
        elif state == 'alert' and dscore <= score_lo:
            st.session_state['drift_state'] = 'recover'
            if _can_toast():
                st.toast("Drift ScoreがLowを下回り: recover に遷移", icon="✅")
                st.session_state['drift_last_toast_ts'] = now_ts
    else:
        if np.isfinite(psi_now):
            now_ts = pd.Timestamp.utcnow()
            last_toast = st.session_state.get('drift_last_toast_ts')
            def _can_toast():
                if not isinstance(last_toast, pd.Timestamp):
                    return True
                return (now_ts - last_toast).total_seconds() >= int(st.session_state.get('drift_alert_cooldown_sec', 120))
            if state in ('ok','recover') and psi_now >= hi_thr:
                st.session_state['drift_state'] = 'alert'
                if auto_pause:
                    st.session_state['enable_trading'] = False
                if _can_toast():
                    st.toast("PSIがHighを超過: alert に遷移", icon="⚠️")
                    st.session_state['drift_last_toast_ts'] = now_ts
            elif state == 'alert' and psi_now <= lo_thr:
                st.session_state['drift_state'] = 'recover'
                if _can_toast():
                    st.toast("PSIがLowを下回り: recover に遷移", icon="✅")
                    st.session_state['drift_last_toast_ts'] = now_ts
    st.write(f"状態: {st.session_state['drift_state']}")
    # θブーストは状態に応じて適用（実際のθ計算側で参照）
    st.session_state['theta_drift_bump_active'] = (st.session_state['drift_state'] == 'alert')
    # 軽量チューナー（重み & キャップ）
    with st.container():
        st.caption("— 総合スコア重み／キャップ（微調整）—")
        w = st.session_state.get('drift_weights', {'psi':0.5,'kl':0.2,'js':0.2,'h':0.1})
        caps = st.session_state.get('drift_caps', {'psi':0.5,'kl':0.5,'js':0.5,'h':1.0})
        cw1, cw2, cw3, cw4 = st.columns(4)
        w['psi'] = cw1.number_input('w_PSI', 0.0, 1.0, float(w.get('psi',0.5)), 0.05)
        w['kl']  = cw2.number_input('w_KL', 0.0, 1.0, float(w.get('kl',0.2)), 0.05)
        w['js']  = cw3.number_input('w_JS', 0.0, 1.0, float(w.get('js',0.2)), 0.05)
        w['h']   = cw4.number_input('w_H',  0.0, 1.0, float(w.get('h',0.1)), 0.05)
        st.session_state['drift_weights'] = w

        cc1, cc2, cc3, cc4 = st.columns(4)
        caps['psi'] = cc1.number_input('cap_PSI', 0.10, 2.00, float(caps.get('psi',0.5)), 0.01)
        caps['kl']  = cc2.number_input('cap_KL',  0.05, 2.00, float(caps.get('kl',0.5)), 0.01)
        caps['js']  = cc3.number_input('cap_JS',  0.05, 2.00, float(caps.get('js',0.5)), 0.01)
        caps['h']   = cc4.number_input('cap_H',   0.10, 2.00, float(caps.get('h',1.0)), 0.01)
        st.session_state['drift_caps'] = caps

        # プリセット適用
        st.caption("— プリセット —")
        p1, p2, p3 = st.columns(3)
        def _apply_drift_preset(name: str):
            if name == "Conservative":
                st.session_state['drift_weights'] = {'psi':0.55,'kl':0.20,'js':0.15,'h':0.10}
                st.session_state['drift_caps']    = {'psi':0.35,'kl':0.30,'js':0.30,'h':0.80}
            elif name == "Balanced":
                st.session_state['drift_weights'] = {'psi':0.50,'kl':0.20,'js':0.20,'h':0.10}
                st.session_state['drift_caps']    = {'psi':0.50,'kl':0.50,'js':0.50,'h':1.00}
            else:  # Aggressive
                st.session_state['drift_weights'] = {'psi':0.45,'kl':0.25,'js':0.20,'h':0.10}
                st.session_state['drift_caps']    = {'psi':0.70,'kl':0.70,'js':0.70,'h':1.20}
            st.toast(f"Driftプリセット適用: {name}")
        with p1:
            if st.button("Conservative", key="drift_preset_consv"):
                _apply_drift_preset("Conservative")
        with p2:
            if st.button("Balanced", key="drift_preset_bal"):
                _apply_drift_preset("Balanced")
        with p3:
            if st.button("Aggressive", key="drift_preset_aggr"):
                _apply_drift_preset("Aggressive")

# --- 直近1–2日 簡易バックテスト（AP/Brier/ECE） ---
with st.sidebar.expander("📊 直近1–2日 簡易評価", expanded=False):
    st.caption("モデル確率と価格ラベルから直近の指標をざっくり確認（未来情報リークなしの特徴量で推論）")
    look_days = st.radio("評価期間", [1, 2], index=0, horizontal=True, key="quick_eval_days")
    n_bins = st.slider("ECEビン数", 5, 20, 10, 1, key="quick_eval_bins")
    run_quick = st.button("▶ 直近評価を実行", key="run_quick_eval")

    if run_quick:
        try:
            # 1) データ取得（優先: 画面の df → フォールバック: data/USDJPY_15m.csv）
            raw = globals().get('df', None)
            if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
                # フォールバック読込
                import os
                csv_path = os.path.join("data", "USDJPY_15m.csv")
                if os.path.exists(csv_path):
                    tmp = pd.read_csv(csv_path)
                    # 列名の正規化
                    ren = {}
                    for c in tmp.columns:
                        lc = c.lower()
                        if lc in ["timestamp","time","date"]:
                            ren[c] = "timestamp"
                        elif lc in ["open","high","low","close","volume"]:
                            ren[c] = lc
                    tmp = tmp.rename(columns=ren)
                    if "timestamp" not in tmp.columns:
                        raise RuntimeError("CSV に timestamp 列が見当たりません")
                    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])  # 可能ならJST
                    # 最低限の列のみ使用
                    keep = [c for c in ["timestamp","open","high","low","close","volume"] if c in tmp.columns]
                    raw = tmp[keep].copy()
                else:
                    raise RuntimeError("価格データが見つかりません（画面dfなし/CSVなし）")

            # 2) 直近 look_days のスライス（余裕を持って過去も含める）
            df_raw = raw.copy()
            if "timestamp" in df_raw.columns:
                df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
                t_end = df_raw["timestamp"].iloc[-1]
                t_start = t_end - pd.Timedelta(days=int(look_days))
                # 特徴量のローリング安定化のため余白 +2日
                t_start_wide = t_start - pd.Timedelta(days=2)
                df_raw = df_raw.loc[df_raw["timestamp"] >= t_start_wide].reset_index(drop=True)
            else:
                # タイムスタンプがない場合は末尾N本で近似
                n_per_day = 96  # 15分足前提
                take = int((2 + int(look_days)) * n_per_day)
                df_raw = df_raw.tail(take).reset_index(drop=True)

            # 3) 特徴量作成（未来情報なし）
            try:
                df_feats = prepare_df_feats_for_inference(df_raw)
            except Exception as e:
                st.error(f"特徴量生成に失敗: {e}")
                df_feats = None
            if df_feats is None or df_feats.empty:
                st.warning("特徴量が空です（データ不足の可能性）")
            else:
                # 4) モデル/メタ読み込み → 推論
                try:
                    model, use_cols = load_break_model("models/break_model.joblib")
                    meta_local = load_break_meta("models/break_meta.json")
                    pred = predict_with_session_theta(df_feats, model, use_cols, meta_local)
                except Exception as e:
                    st.error(f"推論に失敗: {e}")
                    pred = None

                # 5) 教師ラベル（直近H本先のブレイク成否）
                from label_break import build_break_labels, BreakLabelConfig as BreakCfg
                H = 12
                try:
                    H = int(st.session_state.get('break_prob_h', 12))
                except Exception:
                    H = 12
                cfg = BreakCfg(H=H)
                try:
                    windows_df = st.session_state.get('windows_df')
                except Exception:
                    windows_df = None
                try:
                    labels = build_break_labels(df_raw.rename(columns={c:c.lower() for c in df_raw.columns}), cfg,
                                                windows_df=windows_df)
                except Exception as e:
                    st.error(f"ラベル生成に失敗: {e}")
                    labels = None

                # 6) マージ（timestampで内積）
                if pred is not None and labels is not None and not pred.empty and not labels.empty:
                    p = pred.copy()
                    p["timestamp"] = pd.to_datetime(p["timestamp"]).dt.tz_localize(None)
                    y = labels[["timestamp","y"]].copy()
                    y["timestamp"] = pd.to_datetime(y["timestamp"]).dt.tz_localize(None)
                    # 期間フィルタ（本評価期間のみ）
                    if "timestamp" in df_raw.columns:
                        mask = (y["timestamp"] >= t_start)
                        y = y.loc[mask]
                    merged = p.merge(y, on="timestamp", how="inner")
                    merged = merged.dropna(subset=["y","proba"]).copy()
                    merged = merged[(merged["y"].isin([0,1])) & merged["proba"].between(0,1)]

                    if merged.empty:
                        st.warning("評価対象のサンプルがありませんでした。期間や前処理を見直してください。")
                    else:
                        import numpy as _np
                        from sklearn.metrics import average_precision_score, brier_score_loss
                        from calibration import _reliability_table
                        y_true = merged["y"].astype(int).values
                        proba  = merged["proba"].astype(float).values
                        try:
                            ap = float(average_precision_score(y_true, proba))
                        except Exception:
                            ap = float('nan')
                        try:
                            brier = float(brier_score_loss(y_true, proba))
                        except Exception:
                            brier = float('nan')
                        try:
                            cal = _reliability_table(y_true=_np.asarray(y_true), proba=_np.asarray(proba), n_bins=int(n_bins), strategy="quantile")
                            ece = float(cal.ece)
                        except Exception:
                            ece = float('nan')

                        n_all = int(len(merged)); n_pos = int((merged["y"]==1).sum())
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("AP", f"{ap:.4f}" if _np.isfinite(ap) else "N/A")
                            st.caption(f"n={n_all}, pos={n_pos}")
                        with c2:
                            st.metric("Brier", f"{brier:.4f}" if _np.isfinite(brier) else "N/A")
                            st.caption("低いほど良い")
                        with c3:
                            st.metric("ECE", f"{ece:.4f}" if _np.isfinite(ece) else "N/A")
                            st.caption(f"bins={int(n_bins)}")
                        with c4:
                            st.metric("H", f"{H}")

                        # 簡易信頼度プロット（スパークライン風）
                        try:
                            import plotly.graph_objects as go
                            qs = pd.qcut(merged["proba"], q=min(10, max(3,int(n_bins))), duplicates="drop")
                            tab = merged.groupby(qs).agg(prob_mean=("proba","mean"), frac_pos=("y","mean"))
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=tab["prob_mean"], y=tab["frac_pos"], mode="lines+markers", name="Observed"))
                            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Ideal", line=dict(dash="dash")))
                            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=180)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                else:
                    st.warning("推論またはラベルが空のため評価をスキップしました。")
        except Exception as e:
            st.error(f"簡易評価の実行に失敗: {e}")

# --- 簡易バックテスト（グリッド＋Apply Best） ---
with st.sidebar.expander("🧪 グリッド評価（Apply Best）", expanded=False):
    st.caption("θ とドリフト設定（重み/キャップ/ヒステリシス）を小規模グリッドで試し、最良案を適用")

    # — コントロール —
    look_days_g = st.radio("評価期間", [1, 2], index=0, horizontal=True, key="grid_eval_days")
    st.markdown("- θ の探索範囲（base θ を対象）")
    th_min = st.number_input("θ min", 0.30, 0.99, float(st.session_state.get('theta_min', 0.40)), 0.01, key="grid_theta_min")
    th_max = st.number_input("θ max", 0.30, 0.99, float(st.session_state.get('theta_max', 0.85)), 0.01, key="grid_theta_max")
    th_step = st.number_input("θ step", 0.001, 0.10, 0.01, 0.001, key="grid_theta_step")
    tgt_cov = float(st.session_state.get('theta_target_cov', 0.08))
    st.caption(f"目標Coverage={tgt_cov:.2%}（セッション設定から）")

    st.markdown("- ドリフト設定の候補（プリセット×閾値）")
    drift_preset_names = ["Conservative", "Balanced", "Aggressive"]
    hysteresis_sets = [
        {"name": "Tight", "hi": 0.45, "lo": 0.35},
        {"name": "Balanced", "hi": 0.55, "lo": 0.40},
        {"name": "Loose", "hi": 0.65, "lo": 0.50},
    ]
    run_grid = st.button("▶ グリッド評価を実行", key="run_grid_eval")

    def _load_recent_raw_df(days: int) -> pd.DataFrame:
        raw = globals().get('df', None)
        if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
            import os
            csv_path = os.path.join("data", "USDJPY_15m.csv")
            if os.path.exists(csv_path):
                tmp = pd.read_csv(csv_path)
                ren = {}
                for c in tmp.columns:
                    lc = str(c).lower()
                    if lc in ["timestamp","time","date"]:
                        ren[c] = "timestamp"
                    elif lc in ["open","high","low","close","volume"]:
                        ren[c] = lc
                tmp = tmp.rename(columns=ren)
                if "timestamp" not in tmp.columns:
                    return pd.DataFrame()
                tmp["timestamp"] = pd.to_datetime(tmp["timestamp"]).dt.tz_localize(None)
                keep = [c for c in ["timestamp","open","high","low","close","volume"] if c in tmp.columns]
                raw = tmp[keep].copy()
            else:
                return pd.DataFrame()
        df_raw = raw.copy()
        if "timestamp" in df_raw.columns:
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"]).dt.tz_localize(None)
            t_end = df_raw["timestamp"].iloc[-1]
            t_start = t_end - pd.Timedelta(days=int(days))
            t_start_wide = t_start - pd.Timedelta(days=2)
            df_raw = df_raw.loc[df_raw["timestamp"] >= t_start_wide].reset_index(drop=True)
        else:
            n_per_day = 96
            take = int((2 + int(days)) * n_per_day)
            df_raw = df_raw.tail(take).reset_index(drop=True)
        return df_raw

    def _recent_pred_and_labels(days: int) -> pd.DataFrame:
        """直近 days 日の proba と y を返す DataFrame(timestamp, proba, y)。なければ空。"""
        df_raw = _load_recent_raw_df(days)
        if df_raw is None or df_raw.empty:
            return pd.DataFrame()
        try:
            df_feats = prepare_df_feats_for_inference(df_raw)
            model, use_cols = load_break_model("models/break_model.joblib")
            meta_local = load_break_meta("models/break_meta.json")
            pred = predict_with_session_theta(df_feats, model, use_cols, meta_local)
        except Exception:
            return pd.DataFrame()
        from label_break import build_break_labels, BreakLabelConfig as BreakCfg
        H = 12
        try:
            H = int(st.session_state.get('break_prob_h', 12))
        except Exception:
            H = 12
        try:
            labels = build_break_labels(df_raw.rename(columns={c:c.lower() for c in df_raw.columns}), BreakCfg(H=H),
                                        windows_df=st.session_state.get('windows_df'))
        except Exception:
            return pd.DataFrame()
        if pred is None or pred.empty or labels is None or labels.empty:
            return pd.DataFrame()
        p = pred.copy(); y = labels[["timestamp","y"]].copy()
        p["timestamp"] = pd.to_datetime(p["timestamp"]).dt.tz_localize(None)
        y["timestamp"] = pd.to_datetime(y["timestamp"]).dt.tz_localize(None)
        if "timestamp" in df_raw.columns:
            t_end = df_raw["timestamp"].iloc[-1]
            t_start = t_end - pd.Timedelta(days=int(days))
            y = y.loc[y["timestamp"] >= t_start]
        merged = p.merge(y, on="timestamp", how="inner").dropna(subset=["y","proba"])
        merged = merged[(merged["y"].isin([0,1])) & (merged["proba"].between(0,1))]
        return merged

    def _theta_grid_candidates(lo: float, hi: float, step: float) -> list[float]:
        if hi < lo: lo, hi = hi, lo
        n = int(max(1, round((hi - lo) / max(step, 1e-6))) + 1)
        vals = [float(np.round(lo + i*step, 6)) for i in range(n)]
        # 数値誤差で上限を超える場合除去
        return [v for v in vals if lo - 1e-9 <= v <= hi + 1e-9]

    def _drift_preset(name: str) -> tuple[dict, dict]:
        if name == "Conservative":
            return ({'psi':0.55,'kl':0.20,'js':0.15,'h':0.10}, {'psi':0.35,'kl':0.30,'js':0.30,'h':0.80})
        if name == "Aggressive":
            return ({'psi':0.45,'kl':0.25,'js':0.20,'h':0.10}, {'psi':0.70,'kl':0.70,'js':0.70,'h':1.20})
        return ({'psi':0.50,'kl':0.20,'js':0.20,'h':0.10}, {'psi':0.50,'kl':0.50,'js':0.50,'h':1.00})

    def _evaluate_theta(merged_df: pd.DataFrame, thetas: list[float], target_cov: float) -> dict:
        if merged_df is None or merged_df.empty:
            return {"best": None, "table": []}
        rows = []
        n = len(merged_df)
        for th in thetas:
            cov = float((merged_df["proba"] >= th).mean()) if n>0 else float('nan')
            diff = abs(cov - target_cov) if np.isfinite(cov) else float('inf')
            rows.append({"theta": float(th), "coverage": cov, "diff": diff})
        rows = sorted(rows, key=lambda r: (r["diff"], -r["coverage"]))
        best = rows[0] if rows else None
        return {"best": best, "table": rows[:10]}

    def _evaluate_drift_grid(prob_buffer: list[float], baseline_probs: np.ndarray | None,
                             presets: list[str], hysets: list[dict]) -> dict:
        """ドリフトの候補を評価。指標: アラート率（低いほど良い）と平均スコア（低いほど良い）。"""
        if baseline_probs is None or not prob_buffer or len(prob_buffer) < 200:
            return {"best": None, "table": []}
        curr = np.asarray(prob_buffer[-1000:], dtype=float)
        # ローリング窓でドリフトを評価
        W = 300; STEP = 100
        idxs = list(range(max(0, len(curr)-W), len(curr), STEP))
        if not idxs:
            idxs = [len(curr)-W] if len(curr) >= W else [0]
        results = []
        for pname in presets:
            w, caps = _drift_preset(pname)
            for hy in hysets:
                hi, lo = float(hy["hi"]), float(hy["lo"])
                state = 'ok'
                alerts = 0; total = 0; scores = []
                for s0 in idxs:
                    seg = curr[max(0, s0-W):s0] if s0>0 else curr[:W]
                    if seg.size < 100:
                        continue
                    try:
                        # PSI/KL/JS/H
                        psi_val, _edges = compute_psi(baseline_probs, seg, edges=None)
                        ds = drift_summary(baseline_probs, seg)
                        dm = {"psi": float(psi_val),
                              "kl": float(ds.get('kl', float('nan'))),
                              "js": float(ds.get('js', float('nan'))),
                              "hellinger": float(ds.get('hellinger', float('nan')))}
                        score = compute_drift_score(dm, w=w, caps=caps)
                    except Exception:
                        score = float('nan')
                    scores.append(score)
                    # ヒステリシス遷移
                    if state in ('ok','recover') and np.isfinite(score) and score >= hi:
                        state = 'alert'
                    elif state == 'alert' and np.isfinite(score) and score <= lo:
                        state = 'recover'
                    if state == 'alert':
                        alerts += 1
                    total += 1
                alert_rate = (alerts/total) if total>0 else float('nan')
                avg_score = float(np.nanmean(scores)) if scores else float('nan')
                results.append({
                    "preset": pname, "hy": hy["name"], "hi": hi, "lo": lo,
                    "alert_rate": alert_rate, "avg_score": avg_score,
                    "weights": w, "caps": caps
                })
        # ソート: alert率→平均スコア
        results = sorted(results, key=lambda r: (
            float('inf') if not np.isfinite(r["alert_rate"]) else r["alert_rate"],
            float('inf') if not np.isfinite(r["avg_score"]) else r["avg_score"]
        ))
        best = results[0] if results else None
        return {"best": best, "table": results[:6]}

    if run_grid:
        try:
            merged = _recent_pred_and_labels(int(look_days_g))
            thetas = _theta_grid_candidates(float(th_min), float(th_max), float(th_step))
            theta_res = _evaluate_theta(merged, thetas, tgt_cov)

            drift_res = _evaluate_drift_grid(
                st.session_state.get('prob_buffer', []),
                baseline_probs,
                drift_preset_names,
                hysteresis_sets
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**θ Best**")
                b = theta_res.get("best")
                if b:
                    st.write({k:(f"{v:.4f}" if isinstance(v,(int,float)) else v) for k,v in b.items()})
                    st.caption("上位候補（diff小→coverage高）")
                    st.dataframe(pd.DataFrame(theta_res.get("table", [])), use_container_width=True)
                else:
                    st.info("θ 評価に十分なデータがありません。モデル/価格データをご確認ください。")
            with c2:
                st.markdown("**Drift Best**")
                db = drift_res.get("best")
                if db:
                    show = {**{k:db[k] for k in ["preset","hy","hi","lo"]},
                            "alert_rate": (f"{db['alert_rate']*100:.1f}%" if np.isfinite(db['alert_rate']) else "N/A"),
                            "avg_score": (f"{db['avg_score']:.3f}" if np.isfinite(db['avg_score']) else "N/A")}
                    st.write(show)
                    st.caption("上位候補（alert率→平均スコア）")
                    st.dataframe(pd.DataFrame(drift_res.get("table", [])), use_container_width=True)
                else:
                    st.info("ドリフト評価に十分な履歴/ベースラインがありません。")

            # — Apply Best —
            def _apply_with_guards(theta_best: dict | None, drift_best: dict | None):
                st.session_state.setdefault('apply_best_audit', [])
                ts = pd.Timestamp.utcnow()
                entry = {"ts": str(ts)}
                # θ: 1回の変更は±0.05以内にクランプ（安全）
                if theta_best and isinstance(theta_best.get("theta"), (int,float)):
                    old = float(st.session_state.get('theta_base', 0.60))
                    new = float(theta_best["theta"])
                    if abs(new - old) > 0.05:
                        new = old + np.sign(new - old) * 0.05
                    new = float(np.clip(new, float(th_min), float(th_max)))
                    st.session_state['theta_base'] = new
                    entry["theta"] = {"old": old, "new": new}
                # Drift: weights/caps + hysteresis
                if drift_best:
                    w = drift_best.get("weights"); c = drift_best.get("caps")
                    hi = float(drift_best.get("hi", st.session_state.get('drift_score_hi', 0.55)))
                    lo = float(drift_best.get("lo", st.session_state.get('drift_score_lo', 0.40)))
                    # クランプ
                    hi = float(np.clip(hi, 0.30, 0.90)); lo = float(np.clip(lo, 0.20, hi-0.05))
                    if isinstance(w, dict):
                        prev_w = st.session_state.get('drift_weights', {}).copy()
                        st.session_state['drift_weights'] = w
                        entry["drift_weights"] = {"old": prev_w, "new": w}
                    if isinstance(c, dict):
                        prev_c = st.session_state.get('drift_caps', {}).copy()
                        st.session_state['drift_caps'] = c
                        entry["drift_caps"] = {"old": prev_c, "new": c}
                    prev_hi = st.session_state.get('drift_score_hi', 0.55)
                    prev_lo = st.session_state.get('drift_score_lo', 0.40)
                    st.session_state['drift_score_hi'] = hi
                    st.session_state['drift_score_lo'] = lo
                    entry["drift_hysteresis"] = {"old": {"hi": prev_hi, "lo": prev_lo}, "new": {"hi": hi, "lo": lo}}
                    st.session_state['drift_use_composite'] = True
                st.session_state['apply_best_audit'].append(entry)

            colL, colR = st.columns([2,1])
            with colL:
                if st.button("✅ Apply Best を適用", type="primary", key="apply_best_now"):
                    _apply_with_guards(theta_res.get("best"), drift_res.get("best"))
                    st.toast("Best 設定を適用しました", icon="✅")
            with colR:
                if st.button("🧹 監査ログをクリア", key="clear_apply_audit"):
                    st.session_state['apply_best_audit'] = []
                    st.toast("監査ログをクリアしました", icon="🧹")

            # 監査ログの表示
            logs = st.session_state.get('apply_best_audit', [])
            if logs:
                st.caption("— 変更の監査ログ（直近）—")
                st.dataframe(pd.DataFrame(logs[::-1])[:10], use_container_width=True)

        except Exception as e:
            st.error(f"グリッド評価でエラー: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("極値検出（スイング）")
look = st.sidebar.slider("左右の窓幅", 3, 15, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("水平サポレジ（クラスタ）")
eps = st.sidebar.number_input("DBSCAN eps（価格）", value=0.08, step=0.01)
min_samples = st.sidebar.slider("min_samples", 3, 12, 4)

st.sidebar.markdown("---")
st.sidebar.subheader("トレンド＆チャネル")
reg_lookback = st.sidebar.slider("回帰に使う直近本数", 30, 300, 40)
chan_k = st.sidebar.slider("チャネル幅（σの倍率）", 0.5, 3.0, 2.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("判定バッファ")
touch_buffer = st.sidebar.number_input("接触バッファ（価格）", value=0.05, step=0.01)
break_buffer_base = st.sidebar.number_input("ブレイクバッファ（価格）", value=0.05, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("重要度スコアの重み")
w_touch = st.sidebar.slider("接触回数", 0.0, 1.0, 0.30, 0.05)
w_recent = st.sidebar.slider("直近距離（近いほど↑）", 0.0, 1.0, 0.30, 0.05)
w_session = st.sidebar.slider("時間帯（主要セッションで↑）", 0.0, 1.0, 0.20, 0.05)
w_vol = st.sidebar.slider("ボラ（ATR）", 0.0, 1.0, 0.20, 0.05)
w_sum = max(1e-9, w_touch + w_recent + w_session + w_vol)
w_touch, w_recent, w_session, w_vol = [w/w_sum for w in (w_touch, w_recent, w_session, w_vol)]


# ---------- ニュース・指標フィルタ & 赤影 ----------
st.sidebar.markdown("---")
with st.sidebar.expander("ニュース・指標フィルタ / 赤影", expanded=False):
    news_file = st.file_uploader("ニュースCSVをアップロード（任意）", type=["csv"])
    st.caption("受理列: time/timestamp/datetime または date+time、importance[, title]（JST推奨）")

    # フィルタ方式
    news_filter_mode = st.radio(
        "フィルタ方式",
        ["一律±分", "重要度別（赤影と同じ）"],
        index=1, horizontal=True
    )
    news_win = st.slider("一律±分（上を選んだときのみ使用）", 0, 120, 30)
    news_imp_min = st.slider("重要度しきい値 (>=)", 1, 5, 3)

    # 重要度→±分マッピング（赤影/重要度別フィルタで使用）
    st.caption("重要度別ウィンドウ（左右±分）")
    map_5 = st.number_input("★5 → ±分", value=90, step=5)
    map_4 = st.number_input("★4 → ±分", value=30, step=5)
    map_3 = st.number_input("★3 → ±分", value=20, step=5)
    map_2 = st.number_input("★2 → ±分", value=0, step=5)
    map_1 = st.number_input("★1 → ±分", value=0, step=5)
    use_news_shade = st.checkbox("チャートに赤影を重ねて表示", value=True)

    # ハード/ソフト抑制
    apply_news_filter = st.checkbox("ハード抑制（窓内のシグナル無効化）", value=True)
    use_soft_suppress = st.checkbox("ソフト抑制（窓内だけ判定を厳しめに）", value=True)
    soft_break_add = st.number_input("ソフト: ブレイクバッファ 追加", value=0.02, step=0.01, format="%.2f")
    soft_K_add = st.slider("ソフト: K 追加", 0, 10, 4)
    st.markdown("— 自動強化（重要イベントプリセット）—")
    auto_harden = st.checkbox("★4以上の直近イベントで自動ハード化", value=True)
    auto_harden_min_before = st.number_input("ハード化: 直前最低分", value=30, step=5)
    auto_harden_min_after = st.number_input("ハード化: 直後最低分", value=60, step=5)

# ---------------- バックテスト ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("バックテスト")
fwd_n = st.sidebar.slider("ブレイク後 N 本（損益判定）", 5, 120, 20)
spread_pips = st.sidebar.number_input("想定スプレッド（pips）", value=0.5, step=0.1)
run_bt = st.sidebar.button("▶ バックテストを実行")

# ---------------- ブレイク確率 / EV ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("ブレイク確率（今から）")
show_break_prob = st.sidebar.checkbox("今からの水平線ブレイク確率を表示", value=True)
break_prob_topk = st.sidebar.slider("表示する上位ライン数", 1, 20, 10)
break_prob_h = st.sidebar.slider("先読みH（参考表示）", 3, 50, 12)
prob_model_path = st.sidebar.text_input("モデルパス", value="models/break_model.joblib")

st.sidebar.markdown("---")
st.sidebar.subheader("期待値ランキング（確率 × 期待pips）")
show_ev_rank = st.sidebar.checkbox("水平線の期待値ランキングを表示", value=True)
ev_level_min_samples = st.sidebar.slider("レベル別の最低サンプル数", 1, 20, 3)

# ---------- 手動再学習ボタン ----------
st.sidebar.markdown("---")
st.sidebar.subheader("モデルの手動再学習")
proj_dir = str(pathlib.Path(__file__).resolve().parent)
train_script = "ai_train_break.py"
model_path = "models/break_model.joblib"
colA, colB = st.sidebar.columns([1,1])
with colA:
    retrain_now = st.button("再学習を実行", type="primary")
with colB:
    show_log = st.checkbox("ログを表示", value=True)

def _format_ts(ts: float) -> str:
    return pd.Timestamp.fromtimestamp(ts, tz=JST).strftime("%Y-%m-%d %H:%M:%S %Z")

def run_retrain(script_path: str, workdir: str) -> tuple[bool, str]:
    py = sys.executable
    try:
        proc = subprocess.run(
            [py, script_path],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60*20
        )
        ok = proc.returncode == 0
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return ok, log
    except Exception as e:
        return False, f"Exception: {e}"

if retrain_now:
    with st.spinner("学習スクリプトを実行中..."):
        ok, log = run_retrain(train_script, proj_dir)
    model_file = pathlib.Path(proj_dir, model_path)
    if model_file.exists():
        ts = model_file.stat().st_mtime
        st.success(f"学習完了：{model_path} を更新（{_format_ts(ts)}）")
        try:
            st.cache_resource.clear()
        except Exception:
            pass
    else:
        st.error("学習は終了しましたが、モデルファイルが見つかりません。保存パスをご確認ください。")
    if show_log:
        st.subheader("再学習ログ")
        st.code((log or "").strip()[:200000], language="bash")
else:
    mf = pathlib.Path(proj_dir, model_path)
    if mf.exists():
        st.caption(f"現在のモデル更新: {_format_ts(mf.stat().st_mtime)}")
    else:
        st.caption("モデル未作成（先に再学習を実行してください）")



# ---------------- データ取得 ----------------
@st.cache_data(show_spinner=False, ttl=60)
def load_data(sym: str, period: str, interval: str,
              oanda_token: str = "", oanda_account: str = "", oanda_env: str = "practice") -> pd.DataFrame:
    """
    OANDAのAPIキー/Accountが設定されていればOANDAから取得、
    そうでなければyfinanceを使用。
    """
    use_oanda_feed = bool(oanda_token and oanda_account)
    if use_oanda_feed:
        try:
            gran_map = {
                "1m": "M1", "2m": "M2", "5m": "M5", "15m": "M15",
                "30m": "M30", "60m": "H1", "90m": "H1",  # OANDAに90分足はないのでH1にfallback
                "1d": "D"
            }
            gran = gran_map.get(interval, "M15")

            inst = "USD_JPY"

            period_map = {"7d": 7*24*4, "14d": 14*24*4, "30d": 30*24*4,
                          "60d": 60*24*4, "90d": 90*24*4, "180d": 180*24*4, "1y": 365*24*4}
            count = min(period_map.get(period, 2000), 5000)

            base = "https://api-fxpractice.oanda.com" if oanda_env=="practice" else "https://api-fxtrade.oanda.com"
            url = f"{base}/v3/instruments/{inst}/candles"
            headers = {"Authorization": f"Bearer {oanda_token}"}
            params = {"count": count, "granularity": gran, "price": "M", "alignmentTimezone": "UTC"}

            r = requests.get(url, headers=headers, params=params, timeout=10)
            r.raise_for_status()
            candles = r.json().get("candles", [])

            rows = []
            for c in candles:
                if not c.get("complete", False):
                    continue
                t = pd.to_datetime(c["time"])
                rows.append(dict(
                    time=t,
                    open=float(c["mid"]["o"]),
                    high=float(c["mid"]["h"]),
                    low=float(c["mid"]["l"]),
                    close=float(c["mid"]["c"]),
                    volume=int(c.get("volume", 0))
                ))
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows).set_index("time").sort_index()
            df.index = df.index.tz_convert(JST)
            return df

        except Exception as e:
            st.warning(f"OANDAデータ取得失敗: {e} → yfinanceにフォールバックします。")

    # fallback: yfinance
    adj_period = clamp_period_for_interval(period, interval)
    try:
        df = yf.Ticker(sym).history(period=adj_period, interval=interval, auto_adjust=False)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        df = yf.download(sym, period=adj_period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return normalize_ohlcv(df, sym)


with st.spinner("データ取得中..."):
    # 価格データ取得部（fetch_prices/safe_callパターン・実用雛形）
    # サイドバーの設定値を取得
    symbol = st.session_state.get("symbol", "JPY=X")
    period_raw = st.session_state.get("period_raw", "60d")
    interval = st.session_state.get("interval", "15m")
    oanda_token = st.session_state.get("oanda_token", "")
    oanda_account = st.session_state.get("oanda_account", "")
    oanda_env = st.session_state.get("oanda_env", "practice")

    def fetch_prices(symbol, period_raw, interval, oanda_token, oanda_account, oanda_env):
        return load_data(symbol, period_raw, interval, oanda_token, oanda_account, oanda_env)

    df, err = safe_call(fetch_prices, symbol, period_raw, interval, oanda_token, oanda_account, oanda_env)
    if err or df is None or df.empty:
        st.error(f"[price] 取得失敗: {err if err else 'データなし'}")
        st.stop()
    # --- ここでJST統一 ---
    df = ensure_jst_index(df)

# --- ドリフトメタの更新（価格取得後・確率バッファ更新の前後どちらでも可）---
st.session_state.setdefault('prob_buffer', [])
try:
    # ここでは pred 確率が未計算のため、最新価格のダミーで更新せずスキップすることもある
    curr_probs = list(st.session_state.get('prob_buffer', []))[-200:]
    psi_val = float('nan'); sev = 'n/a'; ex_rate=float('nan'); kl=js=hell=float('nan')
    if len(curr_probs) >= 50 and baseline_probs is not None:
        psi_val, sev, ex_rate, kl, js, hell = calc_psi_and_exrate(curr_probs, baseline_probs, st.session_state.get('theta_base',0.6), st.session_state.get('theta_base',0.6))
    st.session_state['drift_meta'] = {
        'psi': psi_val,
        'severity': sev,
        'exceed_rate': ex_rate,
        'kl': kl,
        'js': js,
        'hellinger': hell,
    }
except Exception as e:
    st.session_state['drift_meta'] = st.session_state.get('drift_meta', {})

# ---------------- 極値 & レベル ----------------
def swing_pivots(df: pd.DataFrame, look: int):
    highs = df["high"].rolling(look, center=True).max()
    lows  = df["low"].rolling(look, center=True).min()
    pivot_high = df[(df["high"] == highs)].dropna(subset=["high"])
    pivot_low  = df[(df["low"]  == lows )].dropna(subset=["low"])
    return pivot_high, pivot_low

def horizontal_levels(pivot_high: pd.DataFrame, pivot_low: pd.DataFrame, eps: float, min_samples: int):
    prices = np.r_[pivot_high["high"].values, pivot_low["low"].values].reshape(-1,1)
    if len(prices) == 0: return []
    # epsを自動調整: 過去価格の標準偏差の5%程度を初期値に
    auto_eps = float(np.std(prices)) * 0.05 if eps is None or eps <= 0 else eps
    auto_min_samples = max(3, min_samples)
    labels = DBSCAN(eps=auto_eps, min_samples=auto_min_samples).fit(prices).labels_
    levels = []
    for lab in set(labels) - {-1}:
        lv = prices[labels==lab].mean()
        levels.append(float(lv))
    # 近すぎる水準は間引く
    levels = sorted(set([round(lv, 3) for lv in levels]))
    return levels

pivot_high, pivot_low = swing_pivots(df, look)
levels = horizontal_levels(pivot_high, pivot_low, eps=eps, min_samples=min_samples)

# ---------------- 回帰トレンド & チャネル ----------------
def regression_trend(df: pd.DataFrame, lookback: int, use="low"):
    sub = df.tail(lookback)
    y = sub[use].values
    x = np.arange(len(y))
    if len(x) < 2: return None
    m, b = np.polyfit(x, y, 1)
    t0, t1 = sub.index[0], sub.index[-1]
    y0, y1 = b, m*(len(x)-1) + b
    resid = y - (m*x + b)
    sigma = float(np.std(resid))
    return dict(x0=t0, y0=y0, x1=t1, y1=y1, slope=m, intercept=b, sigma=sigma, n=len(x))

trend = regression_trend(df, reg_lookback, use="low")

# ---------------- レベル重要度スコア ----------------
def compute_level_scores(df: pd.DataFrame, levels: list, touch_buffer: float,
                         w_touch: float, w_recent: float, w_session: float, w_vol: float) -> pd.DataFrame:
    if not levels:
        return pd.DataFrame(columns=["level","score","touches","session_ratio"])
    _atr = atr(df, 14)
    atr_recent = float(_atr.iloc[-1]) if not _atr.empty else 0.0
    rec_close = float(df["close"].iloc[-1])

    import math
    rows=[]
    for lv in levels:
        touch_mask = ((df["low"] <= lv) & (df["high"] >= lv)) | (df["close"].sub(lv).abs() <= touch_buffer)
        touches = int(touch_mask.sum())
        dist = abs(rec_close - lv)
        near = 1.0 / (dist + 1e-6)
        if touches > 0:
            ts_idx = df.index[touch_mask]
            sess_hits = sum(1 for ts in ts_idx if in_sessions(ts) in ("Tokyo","London","NY"))
            session_ratio = sess_hits / touches
        else:
            session_ratio = 0.0
        atr_norm = atr_recent / max(1e-6, rec_close)
        rows.append(dict(level=float(lv), touches=touches, near=near, session_ratio=session_ratio, atr_norm=atr_norm))
    df_sc = pd.DataFrame(rows)
    # 正規化を強化: min-maxだけでなく分散が小さい場合はランク化
    for col in ["touches","near","session_ratio","atr_norm"]:
        colmin, colmax = df_sc[col].min(), df_sc[col].max()
        if math.isclose(colmin, colmax):
            # 差が小さい場合は順位でスコア化
            df_sc[col+"_n"] = df_sc[col].rank(method="average") / len(df_sc)
        else:
            df_sc[col+"_n"] = (df_sc[col]-colmin)/(colmax-colmin)
    df_sc["score"] = (
        w_touch*df_sc["touches_n"] + w_recent*df_sc["near_n"] +
        w_session*df_sc["session_ratio_n"] + w_vol*df_sc["atr_norm_n"]
    ) * 100.0
    return df_sc.sort_values("score", ascending=False)

score_df = compute_level_scores(df, levels, touch_buffer, w_touch, w_recent, w_session, w_vol)

# ---------------- ニュースCSV（堅牢パーサ） ----------------
def parse_news_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame(columns=["time","importance","title"])
    try:
        df = pd.read_csv(file, dtype=str)
    except Exception:
        file.seek(0); df = pd.read_csv(file, dtype=str, header=None)
    df = df.astype(str).fillna("")
    def norm(s: str) -> str:
        s = s.strip().replace("\u3000"," "); s = s.lower()
        return re.sub(r"[\s\-_/()]", "", s)
    TIME_CANDS = {"time","timestamp","datetime","date_time","timejst","datetimejst","日時","日付時刻","発表時刻","発表時間","時刻","時間","date","when"}
    DATE_ONLY = {"date","日付","発表日"}
    CLOCK_ONLY= {"time","時刻","発表時刻","発表時間","時間"}
    IMP_CANDS  = {"importance","重要度","impact","rank","priority","優先度","star","stars"}
    TITLE_CANDS= {"title","イベント","指標名","headline","event","name","内容","subject"}
    orig_cols = list(df.columns)
    norm_cols = [norm(c) for c in orig_cols]
    col_map = dict(zip(norm_cols, orig_cols))
    time_col=imp_col=title_col=None
    for nc in norm_cols:
        if nc in {norm(x) for x in TIME_CANDS} and nc not in {norm(x) for x in DATE_ONLY}:
            time_col = col_map[nc]; break
    if time_col is None:
        date_col = None; clock_col=None
        for nc in norm_cols:
            if nc in {norm(x) for x in DATE_ONLY}: date_col = col_map[nc]
            if nc in {norm(x) for x in CLOCK_ONLY}: clock_col = col_map[nc]
        if date_col and clock_col:
            df["_tmp_time"] = (df[date_col].astype(str).str.strip()+" "+df[clock_col].astype(str).str.strip()).str.strip()
            time_col = "_tmp_time"
    if time_col is None:
        best_col=None; best_ok=-1
        for c in df.columns:
            tryconv = pd.to_datetime(df[c], utc=True, errors="coerce", infer_datetime_format=True)
            ok = tryconv.notna().sum()
            if ok > best_ok and ok>0:
                best_ok = ok; best_col = c
        if best_col is not None: time_col = best_col
    for nc in norm_cols:
        if nc in {norm(x) for x in IMP_CANDS}:
            imp_col = col_map[nc]; break
    if imp_col is None:
        best_col=None; best_numeric=-1
        for c in df.columns:
            if c == time_col: continue
            s = pd.to_numeric(df[c], errors="coerce")
            numeric_ok = s.notna().sum()
            if numeric_ok > best_numeric and numeric_ok>0:
                best_numeric = numeric_ok; best_col=c
        if best_col is not None: imp_col = best_col
    for nc in norm_cols:
        if nc in {norm(x) for x in TITLE_CANDS}:
            title_col = col_map[nc]; break
    if title_col is None:
        cand = [c for c in df.columns if c not in {time_col, imp_col}]
        title_col = cand[0] if cand else None

    if time_col is None or imp_col is None:
        raise ValueError("ニュースCSVに 'time'（または date+time）と 'importance' が必要です。")

    def parse_dt_series(s: pd.Series) -> pd.Series:
        # まずUTCとしてパース
        dt = pd.to_datetime(s, utc=True, errors="coerce", infer_datetime_format=True)
        bad = dt.isna()
        if bad.any():
            fmt_list = ["%Y-%m-%d %H:%M","%Y/%m/%d %H:%M","%Y-%m-%d %H:%M:%S","%Y/%m/%d %H:%M:%S"]
            raw = s[bad].astype(str).str.strip()
            fixed = pd.Series([pd.NaT]*len(raw), index=raw.index)
            for fmt in fmt_list:
                try:
                    parsed = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
                    fixed = fixed.fillna(parsed)
                except Exception:
                    pass
            dt.loc[bad] = fixed
        # タイムゾーン自動判定・柔軟変換
        if getattr(dt.dt, 'tz', None) is None:
            # JSTで記載されている場合（例: 0時～23時のみ）
            hours = dt.dt.hour.dropna()
            if (hours.max() <= 23) and (hours.min() >= 0):
                # JSTとしてローカライズ
                dt = dt.dt.tz_localize("Asia/Tokyo")
            else:
                # UTCとしてローカライズ→JST変換
                dt = dt.dt.tz_localize("UTC").dt.tz_convert("Asia/Tokyo")
        bad = dt.isna()
        if bad.any():
            raw = s[bad].astype(str).str.strip()
            def numparse(x):
                try:
                    v = float(x)
                    if v > 10_000_000_000: return pd.to_datetime(v, unit="ms", utc=True)
                    return pd.to_datetime(v, unit="s", utc=True)
                except Exception:
                    return pd.NaT
            dt.loc[bad] = raw.apply(numparse)
        return dt

    dt_jst = parse_dt_series(df[time_col])
    if dt_jst.notna().sum() == 0:
        return pd.DataFrame()  # 日時解釈できない場合は空DataFrameを返す
    imp = pd.to_numeric(df[imp_col], errors="coerce").fillna(0).astype(int)
    ttl = df[title_col] if (title_col in df.columns) else ""
    out = pd.DataFrame({"time": dt_jst, "importance": imp, "title": ttl}).dropna(subset=["time"])
    return out.sort_values("time").reset_index(drop=True)

news_df = parse_news_csv(news_file)

# アップロードCSVが空または日時解釈不可ならメッセージ表示
if news_file is not None:
    if news_df is None or news_df.empty:
        st.info("本日の主要イベントはありません（またはCSVの日時を解釈できませんでした）")

# ---- 重要度別ウィンドウ生成 & 赤影描画ユーティリティ ----
def build_event_windows(events_df: pd.DataFrame, imp_threshold: int, mapping: dict[int,int]) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=["start","end","importance","title"])
    rows=[]
    for _, r in events_df.iterrows():
        imp = int(r["importance"])
        if imp < imp_threshold: 
            continue
        minutes = mapping.get(imp, 0)
        start = r["time"] - timedelta(minutes=minutes)
        end   = r["time"] + timedelta(minutes=minutes)
        rows.append({"start": start, "end": end, "importance": imp, "title": r.get("title", "")})
    windows = pd.DataFrame(rows)
    return windows.sort_values("start").reset_index(drop=True)

def is_suppressed(ts: pd.Timestamp, windows_df: pd.DataFrame) -> bool:
    if windows_df.empty: return False
    return bool(((windows_df["start"] <= ts) & (ts <= windows_df["end"])).any())

def add_news_shading_to_fig(fig: go.Figure, windows_df: pd.DataFrame) -> go.Figure:
    if windows_df.empty:
        st.warning("赤影ウィンドウが空です。CSV内容・重要度しきい値・日時型をご確認ください。")
        return fig
    color_map = {5:"rgba(255,0,0,0.18)", 4:"rgba(255,0,0,0.12)", 3:"rgba(255,0,0,0.08)"}
    # x軸型に合わせてタイムゾーン統一
    idx_tz = None
    if hasattr(fig, 'data') and len(fig.data) > 0 and hasattr(fig.data[0], 'x'):
        idx = fig.data[0].x
        if hasattr(idx, 'tz'):
            idx_tz = idx.tz
    for _, r in windows_df.iterrows():
        col = color_map.get(int(r["importance"]), "rgba(255,0,0,0.08)")
        x0 = r["start"]
        x1 = r["end"]
        # タイムゾーン統一
        if idx_tz and hasattr(x0, 'tz_convert'):
            x0 = x0.tz_convert(idx_tz)
            x1 = x1.tz_convert(idx_tz)
        fig.add_shape(type="rect", xref="x", x0=x0, x1=x1,
                      yref="paper", y0=0, y1=1, fillcolor=col, line=dict(width=0), layer="below")
    fig.update_layout(
        dragmode="pan",
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(fixedrange=False),
    )
    return fig

# マッピング辞書
imp_map = {5:int(map_5), 4:int(map_4), 3:int(map_3), 2:int(map_2), 1:int(map_1)}
windows_df = pd.DataFrame()
if news_df is not None and not news_df.empty:
    df_w, err = safe_call(build_event_windows, news_df, imp_threshold=news_imp_min, mapping=imp_map)
    if err is None and df_w is not None and {"start","end"}.issubset(df_w.columns):
        windows_df = df_w
    else:
        st.warning(f"[event windows] フォールバックします: {err or '列不足'}")
else:
    st.info("イベント情報なし（抑制は無効）")

# --- 重要イベントの自動強化（ハード化/窓拡大）---
try:
    if 'auto_harden' in locals() and auto_harden and (news_df is not None) and not news_df.empty:
        # 今から見て前後の一定時間は、最低限のハード窓を確保
        now = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) and len(df.index)>0 else pd.Timestamp.now(tz=JST)
        # 対象イベント: 重要度4以上、これから±window内
        near = news_df[news_df['importance'] >= 4]
        if not near.empty:
            before = pd.Timedelta(minutes=int(auto_harden_min_before))
            after  = pd.Timedelta(minutes=int(auto_harden_min_after))
            # 未来・直近のイベントを中心に追加の窓を作る
            rows = []
            for _, r in near.iterrows():
                t = r['time']
                if pd.isna(t):
                    continue
                rows.append({"start": t - before, "end": t + after, "importance": int(r['importance']), "title": r.get('title','')})
            if rows:
                extra = pd.DataFrame(rows)
                if not windows_df.empty:
                    windows_df = pd.concat([windows_df, extra], ignore_index=True)
                else:
                    windows_df = extra
                windows_df = windows_df.sort_values('start').reset_index(drop=True)
                # ハード抑制がOFFでも自動で一時ONにする（事故防止）
                st.session_state['apply_news_filter'] = True
                st.info("重要イベント前後は自動でハード抑制を強化しています（設定で無効可）")
except Exception as e:
    st.warning(f"自動強化処理で問題が発生: {e}")

# --- ニュース設定を session_state に同期（合成θ/サマリで参照） ---
st.session_state['news_win'] = news_win
st.session_state['news_imp_min'] = news_imp_min
st.session_state['news_filter_mode'] = news_filter_mode
st.session_state['use_soft_suppress'] = bool('use_soft_suppress' in locals() and use_soft_suppress)
st.session_state['apply_news_filter'] = bool('apply_news_filter' in locals() and apply_news_filter)
st.session_state['theta_bump_in_news'] = st.session_state.get('theta_bump_in_news', 0.03)
st.session_state['windows_df'] = windows_df

# --- ページ上部にメタ統合サマリを表示 ---
try:
    render_meta_summary(df, windows_df, meta)
except Exception as e:
    st.warning(f"サマリ表示で問題が発生: {e}")

# ---------------- パターン検出（Triangle / Rectangle / Double / Flag / Pennant / H&S） ----------------
@dataclass
class Pattern:
    kind: str
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    params: dict
    quality: float
    direction_bias: str

def _fit_line(xs, ys):
    if len(xs) < 2: 
        return None
    m,b = np.polyfit(xs, ys, 1)
    return m,b


from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

# ========= 共通ユーティリティ =========
def _atr(high, low, close, window=14):
    high = np.asarray(high, dtype=float)
    low  = np.asarray(low,  dtype=float)
    close= np.asarray(close,dtype=float)
    prev_close = np.roll(close, 1)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr[0] = high[0] - low[0]
    alpha = 2.0 / (window + 1.0)
    atr = np.empty_like(tr)
    atr[0] = tr[:window].mean() if len(tr) >= window else tr[0]
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    return atr

def _pivots(series, lb=2, ub=2):
    x = np.asarray(series, dtype=float)
    n = len(x)
    is_max = np.zeros(n, dtype=bool)
    is_min = np.zeros(n, dtype=bool)
    for i in range(lb, n-ub):
        window = x[i-lb:i+ub+1]
        if np.argmax(window) == lb and (window[lb] > window[:lb]).all() and (window[lb] > window[lb+1:]).all():
            is_max[i] = True
        if np.argmin(window) == lb and (window[lb] < window[:lb]).all() and (window[lb] < window[lb+1:]).all():
            is_min[i] = True
    return is_max, is_min

def _fit_line(xs, ys) -> Tuple[float,float,float]:
    if len(xs) < 2: return 0.0, 0.0, 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x0 = x - x.mean()
    s, b = np.polyfit(x0, y, 1)
    intercept = b - s * (-x.mean())
    yhat = s * x0 + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 1e-12 else 0.0
    return float(s), float(intercept), float(r2)

def _line_y(slope, intercept, x):
    return slope * x + intercept

def _norm_slope(slope, price_scale):
    return 0.0 if price_scale <= 0 else slope / float(price_scale)

# ========= Triangle Detector =========
def detect_triangles(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_window: int = 14,
    cons_min_bars: int = 12,
    cons_max_bars: int = 40,
    pivot_lb: int = 2,
    pivot_ub: int = 2,
    flat_tol_norm: float = 0.0012,
    converge_min: float = 0.20,
    width_max_atr: float = 3.5,
    r2_min: float = 0.20,
    parallel_tol: float = 0.22,
    breakout_buffer_atr: float = 0.25,
    confirm_bars: int = 1,
    pretrend_win: int = 24,
    require_breakout: bool = False,
    last_N: int | None = 3000,
    e_step: int = 2,
    len_step: int = 2,
) -> List[Dict]:

    if any(c not in df.columns for c in [high_col, low_col, close_col]):
        raise ValueError("DataFrame must have high/low/close columns")

    # 1) 直近N本だけ見る
    if last_N is not None and len(df) > last_N:
        df = df.iloc[-last_N:]

    highs = df[high_col].to_numpy(dtype=float)
    lows  = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)
    n = len(df)
    if n < max(atr_window + cons_max_bars + 5, 80):
        return []

    atr = _atr(highs, lows, close, window=atr_window)
    is_h, _ = _pivots(highs, pivot_lb, pivot_ub)
    _, is_l = _pivots(lows,  pivot_lb, pivot_ub)
    patterns: List[Dict] = []

    # 事前トレンド計算関数
    def _pretrend_slope(end_idx, win: int) -> float:
        j1 = max(0, end_idx - cons_min_bars)  # 三角開始前付近を狙う
        j0 = max(0, j1 - win)
        if j1 - j0 < 5: return 0.0
        x = np.arange(j1-j0+1, dtype=float)
        y = close[j0:j1+1]
        xm, ym = x.mean(), y.mean()
        den = ((x-xm)**2).sum()
        if den <= 0: return 0.0
        return float(((x-xm)*(y-ym)).sum()/den)

    # ブレイク確定チェック
    def _confirm_break(e_idx: int, dir_side: str, m_up, b_up, m_lo, b_lo) -> Tuple[bool, Optional[int], float, float]:
        """dir_side: 'up' or 'down'; 戻り: (確定?, 確定バーidx, entry_price, stop_suggest)"""
        seq = 0
        j = e_idx
        while j < n:
            up_y = _line_y(m_up, b_up, j)
            lo_y = _line_y(m_lo, b_lo, j)
            thr = breakout_buffer_atr * atr[j]
            c = close[j]
            if dir_side == "up":
                ok = (c >= up_y + thr)
            else:
                ok = (c <= lo_y - thr)
            seq = seq + 1 if ok else 0
            if seq >= max(1, int(confirm_bars)):
                # entry/stop（候補）
                if dir_side == "up":
                    entry = max(up_y + thr, c)  # 終値確定時点の価格を優先
                    stop  = lo_y - 0.25 * atr[j]
                else:
                    entry = min(lo_y - thr, c)
                    stop  = up_y + 0.25 * atr[j]
                return True, j, float(entry), float(stop)
            # 直近数本のみ確認（無限ループ回避）
            if j - e_idx > 3: break
            j += 1
        return False, None, np.nan, np.nan


    # 2) ストライド
    for e in range(atr_window + cons_min_bars, n, e_step):
        best = None  # (quality, dict)
        # 複数長さで探索し、最良だけ採用
        min_pivot_points = 3
        for cons_len in range(cons_min_bars, cons_max_bars+1, len_step):
            s = e - cons_len + 1
            if s < atr_window:
                continue

            # 3) 事前フィルタ（回帰前の早落とし）
            width_raw = highs[s:e+1].max() - lows[s:e+1].min()
            if width_raw > width_max_atr * atr[e] * 1.5:
                continue
            atr_cons = atr[s:e+1].mean(); atr_prev = atr[max(s-14,0):s].mean()
            if atr_cons > atr_prev * 0.95:
                continue

            idxs = np.arange(s, e+1)
            hi_idx = idxs[is_h[s:e+1]]
            lo_idx = idxs[is_l[s:e+1]]

            # ピボットが足りない場合は極値で補完
            if len(hi_idx) < min_pivot_points or len(lo_idx) < min_pivot_points:
                k = min(6, len(idxs))
                if k < min_pivot_points:
                    continue
                top_hi_idx = idxs[np.argsort(highs[s:e+1])[-k:]]
                top_lo_idx = idxs[np.argsort(lows[s:e+1])[:k]]
                hi_idx = np.sort(top_hi_idx[:max(min_pivot_points, len(top_hi_idx)//2)])
                lo_idx = np.sort(top_lo_idx[:max(min_pivot_points, len(top_lo_idx)//2)])
            if len(hi_idx) < min_pivot_points or len(lo_idx) < min_pivot_points:
                continue

            uh_slope, uh_inter, uh_r2 = _fit_line(hi_idx, highs[hi_idx])
            lh_slope, lh_inter, lh_r2 = _fit_line(lo_idx, lows[lo_idx])

            price_scale = close[s:e+1].mean()
            uh_n = _norm_slope(uh_slope, price_scale)
            lh_n = _norm_slope(lh_slope, price_scale)

            # 当てはまり最低限
            if (uh_r2 + lh_r2)/2.0 < r2_min:
                continue

            # 開始幅と終了幅（収束チェック）
            width_s = _line_y(uh_slope, uh_inter, s) - _line_y(lh_slope, lh_inter, s)
            width_e = _line_y(uh_slope, uh_inter, e) - _line_y(lh_slope, lh_inter, e)
            if width_s <= 0 or width_e <= 0:
                continue
            # 幅が馬鹿デカいものを弾く（終盤幅）
            if width_e > width_max_atr * atr[e]:
                continue

            # 収束率（どれだけ狭まったか）: (width_s - width_e)/width_s
            converge = (width_s - width_e) / max(width_s, 1e-9)
            if converge < converge_min:
                continue

            # 型判定（閾値パラメータ化）
            tri_type = None
            flat_thr = flat_tol_norm
            parallel_thr = parallel_tol
            if abs(uh_n) <= flat_thr and lh_n > flat_thr:
                tri_type = "ascending_triangle"
            elif abs(lh_n) <= flat_thr and uh_n < -flat_thr:
                tri_type = "descending_triangle"
            else:
                if np.sign(uh_n) != np.sign(lh_n) and np.sign(uh_n)!=0 and np.sign(lh_n)!=0:
                    rel = abs(abs(uh_n) - abs(lh_n)) / max(abs(uh_n), abs(lh_n), 1e-9)
                    if rel <= parallel_thr:
                        tri_type = "sym_triangle"
            if tri_type is None:
                continue

            # 事前トレンド（上昇/下降/中立で加点）
            pre_slope = _pretrend_slope(e, pretrend_win)
            pre_bias = 0.0
            if tri_type == "ascending_triangle":
                pre_bias = 1.0 if pre_slope > 0 else 0.0
            elif tri_type == "descending_triangle":
                pre_bias = 1.0 if pre_slope < 0 else 0.0
            else:
                pre_bias = 0.5  # 中立

            # ブレイク確定（任意）
            breakout_idx = None
            entry = np.nan
            stop  = np.nan
            target= np.nan
            broken = False
            # 方向仮定
            if tri_type == "ascending_triangle":
                expect = "up"
            elif tri_type == "descending_triangle":
                expect = "down"
            else:
                # シンメトリカルは「直近の終値位置」で暫定方向を仮定
                mid_y = (_line_y(uh_slope, uh_inter, e) + _line_y(lh_slope, lh_inter, e)) * 0.5
                expect = "up" if close[e] >= mid_y else "down"

            # ブレイク確認
            broken, b_idx, entry_cand, stop_cand = _confirm_break(
                e, expect, uh_slope, uh_inter, lh_slope, lh_inter
            )
            if broken:
                breakout_idx = b_idx
                entry = entry_cand
                stop  = stop_cand

            # ターゲット（測定幅＝開始幅をベース）
            height = width_s
            if not np.isnan(entry):
                target = entry + height if expect == "up" else entry - height
            else:
                # 未確定でもライン際の参考値
                up_e = _line_y(uh_slope, uh_inter, e)
                lo_e = _line_y(lh_slope, lh_inter, e)
                entry = up_e if expect == "up" else lo_e
                stop  = lo_e - 0.25*atr[e] if expect == "up" else up_e + 0.25*atr[e]
                target= entry + height if expect == "up" else entry - height

            # 品質スコア（タッチ数・収束率重視）
            fit_q   = max(0.0, min(1.0, (uh_r2 + lh_r2)/2.0))
            conv_q  = max(0.0, min(1.0, (converge - converge_min) / max(1e-9, 1.0 - converge_min)))
            flat_q  = 1.0 - min(1.0, abs(uh_n)/flat_thr) if tri_type=="ascending_triangle" else \
                      1.0 - min(1.0, abs(lh_n)/flat_thr) if tri_type=="descending_triangle" else \
                      1.0 - min(1.0, abs(abs(uh_n)-abs(lh_n))/max(abs(uh_n),abs(lh_n),1e-9))
            # タッチ数で加点（最低3点以上で0.5、5点以上で1.0）
            touch_score = min(len(hi_idx), len(lo_idx))
            touch_q = min(1.0, (touch_score-2)/3.0)
            pre_q   = pre_bias
            quality = float(np.clip(0.25*fit_q + 0.35*conv_q + 0.15*flat_q + 0.15*touch_q + 0.10*pre_q, 0, 1))

            pat = {
                "type": tri_type,
                "dir": "bull" if expect=="up" else "bear",
                "start_idx": int(s),
                "end_idx": int(e),
                "breakout_idx": int(breakout_idx) if breakout_idx is not None else None,
                "upper_line": (float(uh_slope), float(uh_inter), float(uh_r2)),
                "lower_line": (float(lh_slope), float(lh_inter), float(lh_r2)),
                "width_start": float(width_s),
                "width_end": float(width_e),
                "converge_ratio": float(converge),
                "touches_upper": int(len(hi_idx)),
                "touches_lower": int(len(lo_idx)),
                "quality_score": quality,
                "entry": float(entry),
                "stop": float(stop),
                "target": float(target),
            }

            # require_breakoutの挙動明確化
            if require_breakout and breakout_idx is None:
                continue

            # 同一終端eで最良だけ採用
            cand = (quality, pat)
            if best is None or cand[0] > best[0]:
                best = cand

        if best is not None:
            patterns.append(best[1])

    return patterns

def detect_triangles_df(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    pats = detect_triangles(df, **kwargs)
    if not pats:
        return pd.DataFrame(columns=[
            "type","dir","start_idx","end_idx","breakout_idx",
            "upper_slope","upper_intercept","upper_r2",
            "lower_slope","lower_intercept","lower_r2",
            "width_start","width_end","converge_ratio",
            "touches_upper","touches_lower",
            "quality_score","entry","stop","target"
        ])
    rows = []
    for p in pats:
        rows.append([
            p["type"], p["dir"], p["start_idx"], p["end_idx"], p["breakout_idx"],
            p["upper_line"][0], p["upper_line"][1], p["upper_line"][2],
            p["lower_line"][0], p["lower_line"][1], p["lower_line"][2],
            p["width_start"], p["width_end"], p["converge_ratio"],
            p["touches_upper"], p["touches_lower"],
            p["quality_score"], p["entry"], p["stop"], p["target"]
        ])
    cols = ["type","dir","start_idx","end_idx","breakout_idx",
            "upper_slope","upper_intercept","upper_r2",
            "lower_slope","lower_intercept","lower_r2",
            "width_start","width_end","converge_ratio",
            "touches_upper","touches_lower",
            "quality_score","entry","stop","target"]
    return pd.DataFrame(rows, columns=cols)


from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

# ===== 共通ユーティリティ（既に定義済みなら重複可） =====
def _atr(high, low, close, window=14):
    high = np.asarray(high, dtype=float)
    low  = np.asarray(low,  dtype=float)
    close= np.asarray(close,dtype=float)
    prev_close = np.roll(close, 1)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr[0] = high[0] - low[0]
    alpha = 2.0 / (window + 1.0)
    atr = np.empty_like(tr)
    atr[0] = tr[:window].mean() if len(tr) >= window else tr[0]
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    return atr

def _pivots(series, lb=2, ub=2):
    x = np.asarray(series, dtype=float)
    n = len(x)
    is_max = np.zeros(n, dtype=bool)
    is_min = np.zeros(n, dtype=bool)
    for i in range(lb, n-ub):
        window = x[i-lb:i+ub+1]
        if np.argmax(window) == lb and (window[lb] > window[:lb]).all() and (window[lb] > window[lb+1:]).all():
            is_max[i] = True
        if np.argmin(window) == lb and (window[lb] < window[:lb]).all() and (window[lb] < window[lb+1:]).all():
            is_min[i] = True
    return is_max, is_min

def _fit_line(xs, ys) -> Tuple[float,float,float]:
    if len(xs) < 2: return 0.0, 0.0, 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x0 = x - x.mean()
    s, b = np.polyfit(x0, y, 1)
    intercept = b - s * (-x.mean())
    yhat = s * x0 + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 1e-12 else 0.0
    return float(s), float(intercept), float(r2)

def _line_y(slope, intercept, x): return slope * x + intercept
def _norm_slope(slope, price_scale): return 0.0 if price_scale <= 0 else slope / float(price_scale)

# ===== Rectangle Detector =====
def detect_rectangles(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_window: int = 14,
    win_min_bars: int = 10,
    win_max_bars: int = 48,
    pivot_lb: int = 2,
    pivot_ub: int = 2,
    flat_tol_norm: float = 0.0009,     # 「水平」許容の正規化傾き
    drift_tol_norm: float = 0.0007,    # 中心ドリフト（上下線中点の傾き）許容
    width_max_atr: float = 3.5,        # 終盤幅が ATR×この倍以下
    width_stability_max: float = 0.28, # 幅の安定度（std/mean）上限
    min_touches_each: int = 3,         # 各辺の最小タッチ回数
    touch_tol_atr: float = 0.25,       # タッチ判定の許容（ATR倍）
    r2_min: float = 0.12,              # ライン当てはまりの最低R²
    breakout_buffer_atr: float = 0.30, # ブレイク判定のATRバッファ
    confirm_bars: int = 1,             # ブレイク終値の連続確定本数
    require_breakout: bool = False,    # Trueなら確定のみ採用
    # 速度対策
    last_N: Optional[int] = 3000,      # 直近N本に限定
    e_step: int = 1,                   # 終端のステップ
    len_step: int = 1,                 # 窓長のステップ
) -> List[Dict]:
    if any(c not in df.columns for c in [high_col, low_col, close_col]):
        raise ValueError("DataFrame must have high/low/close columns")

    if last_N is not None and len(df) > last_N:
        df = df.iloc[-last_N:].copy()

    highs = df[high_col].to_numpy(dtype=float)
    lows  = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)
    n = len(df)
    if n < max(atr_window + win_max_bars + 5, 80):
        return []

    atr = _atr(highs, lows, close, window=atr_window)
    is_h, _ = _pivots(highs, pivot_lb, pivot_ub)
    _, is_l = _pivots(lows,  pivot_lb, pivot_ub)

    patterns: List[Dict] = []

    def _confirm_break(e_idx: int, side: str, m_up, b_up, m_lo, b_lo):
        """side: 'up'|'down'"""
        seq = 0
        j = e_idx
        while j < n:
            up = _line_y(m_up, b_up, j)
            lo = _line_y(m_lo, b_lo, j)
            thr = breakout_buffer_atr * atr[j]
            c = close[j]
            ok = (c >= up + thr) if side == "up" else (c <= lo - thr)
            seq = seq + 1 if ok else 0
            if seq >= max(1,int(confirm_bars)):
                entry = max(up+thr, c) if side=="up" else min(lo-thr, c)
                stop  = lo - 0.25*atr[j] if side=="up" else up + 0.25*atr[j]
                return True, j, float(entry), float(stop)
            if j - e_idx > 3: break
            j += 1
        return False, None, np.nan, np.nan

    for e in range(atr_window + win_min_bars, n, e_step):
        best = None  # (quality, dict)
        for L in range(win_min_bars, win_max_bars+1, len_step):
            s = e - L + 1
            if s < atr_window: 
                continue
            idxs = np.arange(s, e+1)

            # ピボット抽出
            hi_idx = idxs[is_h[s:e+1]]
            lo_idx = idxs[is_l[s:e+1]]
            # タッチ不足なら極値補完（軽量）
            if len(hi_idx) < 2 or len(lo_idx) < 2:
                k = min(4, len(idxs))
                if k < 2: 
                    continue
                top_hi = idxs[np.argsort(highs[s:e+1])[-k:]]
                bot_lo = idxs[np.argsort(lows[s:e+1])[:k]]
                hi_idx = np.sort(top_hi[:max(2, len(top_hi)//2)])
                lo_idx = np.sort(bot_lo[:max(2, len(bot_lo)//2)])

            # 線当て
            uh_s, uh_b, uh_r2 = _fit_line(hi_idx, highs[hi_idx])
            lh_s, lh_b, lh_r2 = _fit_line(lo_idx, lows[lo_idx])

            price_scale = close[s:e+1].mean()
            uh_n = _norm_slope(uh_s, price_scale)
            lh_n = _norm_slope(lh_s, price_scale)
            mid_n = _norm_slope((uh_s + lh_s)/2.0, price_scale)

            # 水平度と当てはまり
            if abs(uh_n) > flat_tol_norm or abs(lh_n) > flat_tol_norm:
                continue
            if (uh_r2 + lh_r2)/2.0 < r2_min:
                continue
            if abs(mid_n) > drift_tol_norm:
                continue

            # 幅と安定性
            width = _line_y(uh_s, uh_b, idxs) - _line_y(lh_s, lh_b, idxs)
            if np.any(width <= 0):
                continue
            w_mean = float(width.mean())
            w_std  = float(width.std(ddof=0))
            if w_mean > width_max_atr * atr[e]:
                continue
            w_stab = (w_std / max(w_mean, 1e-9))
            if w_stab > width_stability_max:
                continue

            # タッチ判定（ライン±tolに入ったピボット数）
            tol = touch_tol_atr * atr[e]
            up_vals = _line_y(uh_s, uh_b, hi_idx)
            lo_vals = _line_y(lh_s, lh_b, lo_idx)
            touch_up = int(np.sum(np.abs(highs[hi_idx] - up_vals) <= tol))
            touch_lo = int(np.sum(np.abs(lows[lo_idx]  - lo_vals) <= tol))
            if touch_up < min_touches_each or touch_lo < min_touches_each:
                continue

            # 期待方向は未確定。直近の位置で仮定（上半分→up、下半分→down）
            up_e = float(_line_y(uh_s, uh_b, e))
            lo_e = float(_line_y(lh_s, lh_b, e))
            mid_e = (up_e + lo_e) * 0.5
            expect = "up" if close[e] >= mid_e else "down"

            # ブレイク確認（任意）
            broken, b_idx, entry, stop = _confirm_break(e, expect, uh_s, uh_b, lh_s, lh_b)
            if require_breakout and not broken:
                continue

            # ターゲット＝レンジ高（開始幅 or 平均幅）
            height = float(width[0])  # 開始幅
            if np.isnan(entry):
                # 未確定でも参考値（ライン際）
                entry = up_e if expect=="up" else lo_e
                stop  = lo_e - 0.25*atr[e] if expect=="up" else up_e + 0.25*atr[e]
            target = entry + height if expect=="up" else entry - height

            # 品質スコア（0–1）
            fit_q   = max(0.0, min(1.0, (uh_r2 + lh_r2)/2.0))
            flat_q  = 1.0 - min(1.0, max(abs(uh_n), abs(lh_n)) / flat_tol_norm)
            stab_q  = 1.0 - min(1.0, w_stab / width_stability_max)
            touch_q = min(1.0, 0.5*min(1.0, touch_up/min_touches_each) + 0.5*min(1.0, touch_lo/min_touches_each))
            drift_q = 1.0 - min(1.0, abs(mid_n)/drift_tol_norm)
            quality = float(np.clip(0.30*fit_q + 0.25*flat_q + 0.20*stab_q + 0.15*touch_q + 0.10*drift_q, 0, 1))

            pat = {
                "type": "rectangle",
                "dir": "bull" if expect=="up" else "bear",  # 期待方向（暫定／確定で上書き可）
                "start_idx": int(s),
                "end_idx": int(e),
                "breakout_idx": int(b_idx) if b_idx is not None else None,
                "upper_line": (float(uh_s), float(uh_b), float(uh_r2)),
                "lower_line": (float(lh_s), float(lh_b), float(lh_r2)),
                "width_mean": float(w_mean),
                "width_std": float(w_std),
                "width_stability": float(w_stab),
                "touches_upper": int(touch_up),
                "touches_lower": int(touch_lo),
                "quality_score": quality,
                "entry": float(entry), "stop": float(stop), "target": float(target),
            }

            cand = (quality, pat)
            if best is None or cand[0] > best[0]:
                best = cand

        if best is not None:
            patterns.append(best[1])

    return patterns

def detect_rectangles_df(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    pats = detect_rectangles(df, **kwargs)
    if not pats:
        return pd.DataFrame(columns=[
            "type","dir","start_idx","end_idx","breakout_idx",
            "upper_slope","upper_intercept","upper_r2",
            "lower_slope","lower_intercept","lower_r2",
            "width_mean","width_std","width_stability",
            "touches_upper","touches_lower",
            "quality_score","entry","stop","target"
        ])
    rows = []
    for p in pats:
        rows.append([
            p["type"], p["dir"], p["start_idx"], p["end_idx"], p["breakout_idx"],
            p["upper_line"][0], p["upper_line"][1], p["upper_line"][2],
            p["lower_line"][0], p["lower_line"][1], p["lower_line"][2],
            p["width_mean"], p["width_std"], p["width_stability"],
            p["touches_upper"], p["touches_lower"],
            p["quality_score"], p["entry"], p["stop"], p["target"]
        ])
    cols = ["type","dir","start_idx","end_idx","breakout_idx",
            "upper_slope","upper_intercept","upper_r2",
            "lower_slope","lower_intercept","lower_r2",
            "width_mean","width_std","width_stability",
            "touches_upper","touches_lower",
            "quality_score","entry","stop","target"]
    return pd.DataFrame(rows, columns=cols)

def detect_double_top_bottom(
    df, piv_high, piv_low, *,
    lookback=200,
    # === 推奨デフォルト(USDJPY 15m) ===
    tol_mode="atr",       # "pct" or "atr"（ボラ正規化）
    tol_pct=0.10,
    tol_atrK=0.55,        # 2山/2底の同値許容（ATR倍）
    min_sep_bars=10,      # ピーク間の最低バー数
    min_depth_atr=0.90,   # M/W の谷/山の最小深さ（ATR倍）
    confirm_bars=2,       # ネック終値ブレイクの連続本数
    neck_break_atr=0.15,  # ネック抜け判定（ATR倍）
    retest_within=30,     # ブレイク後のリテスト探索窓
    retest_tol_atr=0.25,  # リテストはネック±ATR*係数内
    pretrend_win=24,      # 事前トレンド傾き算出の窓（15m×24=約6h）
    pretrend_min=0.0,     # 0以上→topは上昇/ bottomは下降を優先
):
    import numpy as np, pandas as pd

    if len(df) < max(40, lookback // 2):
        return []

    # ---- 対象窓（Index前提）----
    sub = df.tail(lookback).copy()
    idx = sub.index
    c = sub["close"].astype(float)
    h = sub["high"].astype(float)
    l = sub["low"].astype(float)

    # ---- ATR14（fallback付き）----
    def _atr14(_df):
        _h, _l, _c = _df["high"].astype(float), _df["low"].astype(float), _df["close"].astype(float)
        pc = _c.shift(1)
        tr = pd.concat([(_h - _l).abs(), (_h - pc).abs(), (_l - pc).abs()], axis=1).max(axis=1)
        return tr.rolling(14, min_periods=8).mean()
    a14 = _atr14(sub).ffill()
    a14_med = float(a14.median()) if a14.notna().any() else float((h - l).abs().rolling(14).mean().iloc[-1])

    # ---- 単回帰傾き（事前トレンド）----
    def _slope(s, win):
        x = np.arange(win, dtype=float)
        def _sl(arr):
            y = np.asarray(arr, float)
            xm, ym = x.mean(), y.mean()
            den = ((x - xm) ** 2).sum()
            if den <= 0: return 0.0
            return float(((x - xm) * (y - ym)).sum() / den)
        return s.rolling(win, min_periods=win).apply(_sl, raw=True)
    slope = _slope(c, pretrend_win)

    def _eq_peak(v1, v2):
        if tol_mode == "atr":
            return abs(v1 - v2) <= float(tol_atrK) * a14_med
        base = max(1e-9, (abs(v1) + abs(v2)) * 0.5)
        return abs(v1 - v2) <= base * float(tol_pct)

    def _confirm_break(i2, neck, side: str):
        # i2(右肩bar) 以降で終値が neck±(ATR*neck_break_atr) を連続 confirm_bars 本
        thr = float(neck_break_atr) * a14_med
        seg = sub.loc[i2:]
        if seg.empty: return (False, None)
        cc = seg["close"].astype(float)
        if side == "down":
            hit = (cc <= (neck - thr)).rolling(confirm_bars).sum() >= confirm_bars
        else:
            hit = (cc >= (neck + thr)).rolling(confirm_bars).sum() >= confirm_bars
        if not hit.any(): return (False, None)
        # 最初に条件を満たした行ラベル（Indexラベル）を返す
        j = hit[hit].index[0]
        return (True, j)

    def _retest_after(brk_label, neck, tol_atr, within):
        if brk_label is None: return (False, None)
        start_pos = idx.get_loc(brk_label) + 1
        end_pos = min(len(idx) - 1, start_pos + int(within))
        if start_pos >= end_pos: return (False, None)
        seg = sub.iloc[start_pos:end_pos + 1]
        tol = float(tol_atr) * a14_med
        hit = seg[(seg["high"] >= neck - tol) & (seg["low"] <= neck + tol)]
        if hit.empty: return (False, None)
        return (True, hit.index[0])

    # ===== 品質スコア（重みはここで調整）=====
    def _quality(eq_score, depth_z, gap_bars, confirmed, retested, trend_ok):
        q = 50.0
        q += 20.0 * max(0.0, min(1.0, eq_score))             # 2山/2底の同値度（最重要）
        q += min(20.0, 6.0 * max(0.0, depth_z - 0.5))        # 深さ（0.5ATR超から厚めに加点）
        q += 8.0 if confirmed else 0.0                       # ネック終値確定
        q += 7.0 if retested  else 0.0                       # リテスト確認
        q += 5.0 if trend_ok  else 0.0                       # 事前トレンド整合
        # ピーク間隔：12〜28本が理想、外れるほど減点（最大10）
        q -= min(10.0, abs(int(gap_bars) - 20) * 0.4)
        return float(max(0.0, min(99.0, q)))

    out = []

    # -------- Double Top --------
    highs = piv_high.index.intersection(idx).sort_values()
    for i in range(len(highs) - 1):
        i1, i2 = highs[i], highs[i + 1]
        gap = idx.get_loc(i2) - idx.get_loc(i1)
        if gap < int(min_sep_bars): 
            continue
        h1, h2 = float(df.loc[i1, "high"]), float(df.loc[i2, "high"])
        if not _eq_peak(h1, h2):
            continue
        span = sub.loc[i1:i2]
        neck = float(span["low"].min())
        topv = (h1 + h2) * 0.5
        depth = topv - neck
        if depth < float(min_depth_atr) * a14_med:
            continue

        confirmed, c_lab = _confirm_break(i2, neck, side="down")
        retested, r_lab  = _retest_after(c_lab if confirmed else i2, neck, retest_tol_atr, retest_within)

        sl = float(slope.loc[i1]) if not np.isnan(slope.loc[i1]) else 0.0
        trend_ok = (sl > float(pretrend_min))

        eq_score = 1.0 - (abs(h1 - h2) / max(1e-9, abs(topv)))
        depth_z  = depth / max(1e-9, a14_med)
        q = _quality(eq_score, depth_z, gap, confirmed, retested, trend_ok)

        out.append(Pattern(
            "double_top",
            t_start=i1, t_end=i2,
            params={"top": float(topv), "neck": float(neck),
                    "confirmed": bool(confirmed), "confirm_ts": c_lab,
                    "retested": bool(retested), "retest_ts": r_lab},
            quality=q,
            direction_bias="down"
        ))

    # -------- Double Bottom --------
    lows = piv_low.index.intersection(idx).sort_values()
    for i in range(len(lows) - 1):
        i1, i2 = lows[i], lows[i + 1]
        gap = idx.get_loc(i2) - idx.get_loc(i1)
        if gap < int(min_sep_bars):
            continue
        l1, l2 = float(df.loc[i1, "low"]), float(df.loc[i2, "low"])
        if not _eq_peak(l1, l2):
            continue
        span = sub.loc[i1:i2]
        neck = float(span["high"].max())
        botv = (l1 + l2) * 0.5
        depth = neck - botv
        if depth < float(min_depth_atr) * a14_med:
            continue

        confirmed, c_lab = _confirm_break(i2, neck, side="up")
        retested, r_lab  = _retest_after(c_lab if confirmed else i2, neck, retest_tol_atr, retest_within)

        sl = float(slope.loc[i1]) if not np.isnan(slope.loc[i1]) else 0.0
        trend_ok = (sl < -float(pretrend_min))

        eq_score = 1.0 - (abs(l1 - l2) / max(1e-9, abs(botv)))
        depth_z  = depth / max(1e-9, a14_med)
        q = _quality(eq_score, depth_z, gap, confirmed, retested, trend_ok)

        out.append(Pattern(
            "double_bottom",
            t_start=i1, t_end=i2,
            params={"bottom": float(botv), "neck": float(neck),
                    "confirmed": bool(confirmed), "confirm_ts": c_lab,
                    "retested": bool(retested), "retest_ts": r_lab},
            quality=q,
            direction_bias="up"
        ))

    return out


def detect_flag_pennant(df, lookback=220, Npush=30, min_flag_bars=8, max_flag_bars=40, sigma_k=1.0, pole_min_atr=2.0):
    if len(df) < Npush + max_flag_bars + 5: return []
    sub = df.tail(lookback)
    cons = sub.tail(max_flag_bars)
    pole = sub.iloc[-(max_flag_bars+Npush):-max_flag_bars]
    if len(pole) < 5 or len(cons) < min_flag_bars: return []

    pole_len = float(pole["close"].iloc[-1] - pole["close"].iloc[0])
    from flag_pennant_detector import detect_flag_pennant
    pole_dir = "up" if pole_len > 0 else "down"
    pole_abs = abs(pole_len)

    atr_now = float(atr(sub, 14).iloc[-1]) if len(sub)>=14 else 0.0
    if atr_now <= 0: return []
    if pole_abs < pole_min_atr * atr_now:
        return []

    y = cons["close"].values
    x = np.arange(len(y))
    m, b = np.polyfit(x, y, 1)
    resid = y - (m*x + b)
    sigma = float(np.std(resid))
    if sigma <= 0: return []
    x_now = len(x)-1
    y_mid_now = m*x_now + b
    up_now = y_mid_now + sigma_k*sigma
    dn_now = y_mid_now - sigma_k*sigma

    kind = "pennant" if abs(m) < 1e-4 else ("flag_up" if m>0 else "flag_dn")
    direction_bias = "up" if pole_dir=="up" else "down"
    quality = 65.0
    quality += min(20.0, pole_abs/atr_now*3.0)
    quality += min(10.0, (1.0/(sigma/atr_now+1e-6)))

    return [Pattern(
        kind=kind,
        t_start=cons.index[0], t_end=cons.index[-1],
        params=dict(slope=m, intercept=b, sigma=sigma, band_k=float(sigma_k),
                    sub_start=cons.index[0], sub_end=cons.index[-1],
                    pole=float(pole_len), pole_abs=float(pole_abs),
                    upper_now=float(up_now), lower_now=float(dn_now)),
        quality=float(quality),
        direction_bias=direction_bias
    )]

def detect_head_shoulders(
    df: pd.DataFrame,
    piv_high: pd.DataFrame,
    piv_low: pd.DataFrame,
    *,
    lookback: int = 260,
    tol: float = 0.003,          # UIのhs_tol（肩の比率許容）。ATR基準と“OR”で緩い方を採用
    # 追加パラメータ（デフォルトはUSDJPY 15m向け）
    min_sep_bars: int = 10,      # 左肩-頭 / 頭-右肩 の最小バー間隔
    shoulder_tol_atr: float = 0.60,  # 肩の同値許容（ATR倍）
    head_margin_atr: float = 0.80,   # ヘッドが肩よりどれだけ突出しているか（ATR倍）
    confirm_bars: int = 2,       # ネック終値ブレイクの連続本数
    neck_break_atr: float = 0.15,# ネック抜け判定のATR倍
    retest_within: int = 30,     # 確定後、何本以内にリテスト探索
    retest_tol_atr: float = 0.25,# リテスト時のネック±許容（ATR倍）
    pretrend_win: int = 24,      # 事前トレンド評価窓（バー）
    pretrend_min: float = 0.0,   # トップ型は +min 以上、逆H&Sは -min 以下を好評価
    allow_incomplete: bool = False  # Trueなら未確定でも候補化（品質は下げる）
) -> list[Pattern]:
    import numpy as np, pandas as pd

    out: list[Pattern] = []

    if len(df) < max(40, lookback // 2):
        return out

    sub = df.tail(lookback).copy()
    idx = sub.index
    close = sub["close"].astype(float)
    high  = sub["high"].astype(float)
    low   = sub["low"].astype(float)

    # --- ATR14（fallback付き） ---
    def _atr14(_df):
        _h, _l, _c = _df["high"].astype(float), _df["low"].astype(float), _df["close"].astype(float)
        pc = _c.shift(1)
        tr = pd.concat([(_h-_l).abs(), (_h-pc).abs(), (_l-pc).abs()], axis=1).max(axis=1)
        return tr.rolling(14, min_periods=8).mean()
    a14 = _atr14(sub).ffill()
    a14_med = float(a14.median()) if a14.notna().any() else float((high - low).abs().rolling(14).mean().iloc[-1] or 0.0)
    if a14_med <= 0:
        # スケールが取れない時は価格比で最低限動く
        a14_med = float(max(1e-6, (high.tail(20).max() - low.tail(20).min()) / 20.0))

    # --- ユーティリティ ---
    idx_pos = {t:i for i,t in enumerate(idx)}

    def _lin_neck(x1, y1, x2, y2):
        """2点からネック直線 m,b を返す（xはバー番号）"""
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        if x2 == x1: 
            return 0.0, float((y1+y2)/2.0)
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b

    def _confirm_break(side: str, i_start_label) -> tuple[bool, pd.Timestamp|None]:
        """side: 'down' or 'up'（H&S=down, 逆H&S=up）。右肩確定以降で連続本確定を探す。"""
        j0 = idx.get_loc(i_start_label)
        seq, hit_ts = 0, None
        for j in range(j0, len(idx)):
            ts = idx[j]
            c = float(close.loc[ts])
            # 動的ネック値
            neck_val = float(m_neck * j + b_neck)
            thr = neck_break_atr * a14_med
            if side == "down":
                ok = (c <= neck_val - thr)
            else:
                ok = (c >= neck_val + thr)
            seq = seq + 1 if ok else 0
            if seq >= int(confirm_bars):
                hit_ts = ts
                break
        return (hit_ts is not None), hit_ts

    def _retest_after(confirm_label, within: int, tol_atr: float, side: str) -> tuple[bool, pd.Timestamp|None]:
        if confirm_label is None:
            return (False, None)
        j0 = idx.get_loc(confirm_label) + 1
        j1 = min(len(idx)-1, j0 + int(within))
        tol = float(tol_atr) * a14_med
        for j in range(j0, j1+1):
            ts = idx[j]
            neck_val = float(m_neck * j + b_neck)
            # 高安どちらかがネック±tol に触れればリテスト成立
            if (float(high.loc[ts]) >= neck_val - tol) and (float(low.loc[ts]) <= neck_val + tol):
                return (True, ts)
        return (False, None)

    def _pretrend_slope(end_label, win: int) -> float:
        j1 = idx.get_loc(end_label)
        j0 = max(0, j1 - int(win))
        if j1 - j0 < 3:
            return 0.0
        x = np.arange(j1-j0+1, dtype=float)
        y = close.iloc[j0:j1+1].values.astype(float)
        xm, ym = x.mean(), y.mean()
        den = ((x-xm)**2).sum()
        if den <= 0:
            return 0.0
        return float(((x-xm)*(y-ym)).sum()/den)

    # ========= 1) トップ型（H&S） =========
    # ピボット点の間引き（近接点除外）
    def thin_pivots(piv_idx, min_dist=5):
        thinned = []
        last = None
        for t in piv_idx:
            if last is None or abs(idx_pos[t] - idx_pos[last]) >= min_dist:
                thinned.append(t)
                last = t
        return pd.Index(thinned)

    hs = thin_pivots(piv_high.index.intersection(idx).sort_values(), min_dist=5)
    if len(hs) >= 3:
        for k in range(len(hs)-2):
            s1, h, s2 = hs[k], hs[k+1], hs[k+2]
            if (idx_pos[s2] - idx_pos[h] < min_sep_bars) or (idx_pos[h] - idx_pos[s1] < min_sep_bars):
                continue

            v1, vh, v2 = float(df.loc[s1,"high"]), float(df.loc[h,"high"]), float(df.loc[s2,"high"])

            same_shoulder_ok = abs(v1 - v2) <= max(tol*max(vh,1e-9), shoulder_tol_atr*a14_med)
            head_margin_ok   = vh >= max(v1, v2) + head_margin_atr*a14_med
            if not (same_shoulder_ok and head_margin_ok):
                continue

            left_valley  = float(df.loc[s1:h, "low"].min())
            right_valley = float(df.loc[h:s2, "low"].min())
            x1, y1 = idx_pos[s1], left_valley
            x2, y2 = idx_pos[s2], right_valley
            m_neck, b_neck = _lin_neck(x1, y1, x2, y2)

            confirmed, c_lab = _confirm_break(side="down", i_start_label=s2)
            retested, r_lab = _retest_after(c_lab if confirmed else s2, retest_within, retest_tol_atr, side="down")
            sl = _pretrend_slope(s1, pretrend_win)
            pretrend_ok = (sl > float(pretrend_min))
            neck_center = (y1 + y2) * 0.5
            depth_z = max(0.0, (vh - neck_center) / max(1e-9, a14_med))
            gapL = idx_pos[h]  - idx_pos[s1]
            gapR = idx_pos[s2] - idx_pos[h]
            time_sym = 1.0 - abs(gapL - gapR) / max(1.0, max(gapL, gapR))
            neck_slope_penalty = min(12.0, abs(m_neck) / max(1e-9, a14_med) * 6.0)
            q = 45.0
            q += 20.0 * max(0.0, 1.0 - abs(v1 - v2) / max(1e-6, max(v1,v2)))
            q += min(18.0, 6.0 * max(0.0, depth_z - 0.5))
            q += 8.0 if confirmed else (2.0 if allow_incomplete else 0.0)
            q += 6.0 if retested  else 0.0
            q += 5.0 if pretrend_ok else 0.0
            q += 5.0 * max(0.0, time_sym)
            q -= neck_slope_penalty
            q = float(max(0.0, min(99.0, q)))

            out.append(Pattern(
                kind="head_shoulders",
                t_start=s1, t_end=s2,
                params={
                    "head": float(vh),
                    "left": float(v1),
                    "right": float(v2),
                    "neck": float(neck_center),
                    "neck_left": float(y1),
                    "neck_right": float(y2),
                    "neck_m": float(m_neck),
                    "neck_b": float(b_neck),
                    "confirmed": bool(confirmed),
                    "confirm_ts": c_lab,
                    "retested": bool(retested),
                    "retest_ts": r_lab,
                    "left_shoulder_x": s1,
                    "left_shoulder_y": float(v1),
                    "head_x": h,
                    "head_y": float(vh),
                    "right_shoulder_x": s2,
                    "right_shoulder_y": float(v2),
                    "neck_left_x": s1,
                    "neck_left_y": float(y1),
                    "neck_right_x": s2,
                    "neck_right_y": float(y2),
                },
                quality=q,
                direction_bias="down"
            ))

    # ========= 2) 逆H&S（インバース） =========
    ls = piv_low.index.intersection(idx).sort_values()
    if len(ls) >= 3:
        for k in range(len(ls)-2):
            s1, h, s2 = ls[k], ls[k+1], ls[k+2]
            if (idx_pos[s2] - idx_pos[h] < min_sep_bars) or (idx_pos[h] - idx_pos[s1] < min_sep_bars):
                continue

            v1, vh, v2 = float(df.loc[s1,"low"]), float(df.loc[h,"low"]), float(df.loc[s2,"low"])

            same_shoulder_ok = abs(v1 - v2) <= max(tol*max(abs(vh),1e-9), shoulder_tol_atr*a14_med)
            head_margin_ok   = vh <= min(v1, v2) - head_margin_atr*a14_med
            if not (same_shoulder_ok and head_margin_ok):
                continue

            left_peak  = float(df.loc[s1:h, "high"].max())
            right_peak = float(df.loc[h:s2, "high"].max())
            x1, y1 = idx_pos[s1], left_peak
            x2, y2 = idx_pos[s2], right_peak
            m_neck, b_neck = _lin_neck(x1, y1, x2, y2)

            confirmed, c_lab = _confirm_break(side="up", i_start_label=s2)
            retested, r_lab  = _retest_after(c_lab if confirmed else s2, retest_within, retest_tol_atr, side="up")

            sl = _pretrend_slope(s1, pretrend_win)
            pretrend_ok = (sl < -float(pretrend_min))

            neck_center = (y1 + y2) * 0.5
            depth_z = max(0.0, (neck_center - vh) / max(1e-9, a14_med))

            gapL = idx_pos[h]  - idx_pos[s1]
            gapR = idx_pos[s2] - idx_pos[h]
            time_sym = 1.0 - abs(gapL - gapR) / max(1.0, max(gapL, gapR))
            neck_slope_penalty = min(12.0, abs(m_neck) / max(1e-9, a14_med) * 6.0)

            q = 45.0
            q += 20.0 * max(0.0, 1.0 - abs(v1 - v2) / max(1e-6, max(abs(v1),abs(v2))))
            q += min(18.0, 6.0 * max(0.0, depth_z - 0.5))
            q += 8.0 if confirmed else (2.0 if allow_incomplete else 0.0)
            q += 6.0 if retested  else 0.0
            q += 5.0 if pretrend_ok else 0.0
            q += 5.0 * max(0.0, time_sym)
            q -= neck_slope_penalty
            q = float(max(0.0, min(99.0, q)))

            out.append(Pattern(
                kind="inverse_head_shoulders",
                t_start=s1, t_end=s2,
                params={
                    "head": float(vh),
                    "left": float(v1),
                    "right": float(v2),
                    "neck": float(neck_center),
                    "neck_left": float(y1),
                    "neck_right": float(y2),
                    "neck_m": float(m_neck),
                    "neck_b": float(b_neck),
                    "confirmed": bool(confirmed),
                    "confirm_ts": c_lab,
                    "retested": bool(retested),
                    "retest_ts": r_lab,
                    "left_shoulder_x": s1,
                    "left_shoulder_y": float(v1),
                    "head_x": h,
                    "head_y": float(vh),
                    "right_shoulder_x": s2,
                    "right_shoulder_y": float(v2),
                    "neck_left_x": s1,
                    "neck_left_y": float(y1),
                    "neck_right_x": s2,
                    "neck_right_y": float(y2),
                },
                quality=q,
                direction_bias="up"
            ))

    return out


def measured_targets(p: Pattern):
    """パターン別ターゲット（測定値）"""
    kind = p.get('kind') if isinstance(p, dict) else getattr(p, 'kind', None)
    params = p['params'] if isinstance(p, dict) else getattr(p, 'params', {})
    if kind and kind.startswith("triangle"):
        t = float(params.get("thickness", 0.0))
        return dict(up=+t, down=-t)
    if kind=="rectangle":
        h = float(params.get("height", 0.0))
        return dict(up=+h, down=-h)
    if kind in ("double_top","double_bottom"):
        neck = params.get("neck")
        ref  = params.get("top", params.get("bottom", None))
        if neck is not None and ref is not None:
            h = abs(float(ref) - float(neck))
            return dict(up=+h, down=-h)
    if kind in ("flag_up","flag_dn","pennant"):
        pole = float(params.get("pole_abs", 0.0))
        return dict(up=+pole, down=-pole)
    if kind in ("head_shoulders","inverse_head_shoulders"):
        neck = params.get("neck")
        head = params.get("head")
        if neck is None or head is None:
            return dict(up=0.0, down=0.0)
        try:
            neck = float(neck)
            head = float(head)
            h = abs(head - neck)
            return dict(up=+h, down=-h)
        except Exception:
            return dict(up=0.0, down=0.0)
    return dict(up=0.0, down=0.0)

# ---- パターン検出設定（サイドバー）----
st.sidebar.markdown("---")
st.sidebar.subheader("パターン検出")
enable_tri    = st.sidebar.checkbox("トライアングル（対称/上昇/下降）", True)
enable_rect   = st.sidebar.checkbox("レクタングル（ボックス）", True)
enable_double = st.sidebar.checkbox("ダブルトップ / ダブルボトム", True)
enable_flag   = st.sidebar.checkbox("フラッグ / ペナント", True)
enable_hs     = st.sidebar.checkbox("ヘッド＆ショルダーズ（逆含む）", True)

st.sidebar.subheader("パターンごとの直近本数")
tri_lookback = st.sidebar.slider("トライアングル（対称/上昇/下降）", 20, 600, 125, 5)
rect_lookback = st.sidebar.slider("レクタングル（ボックス）", 20, 600, 150, 5)
double_lookback = st.sidebar.slider("ダブルトップ/ダブルボトム", 20, 600, 75, 5)
# — ダブルトップ／ダブルボトム 設定 —
st.sidebar.caption("— ダブルトップ／ダブルボトム 設定 —")
dbl_tol_mode = st.sidebar.selectbox("許容幅の基準", ["atr(ATR倍)", "pct(％)"], index=0)
dbl_tol_atrK = st.sidebar.number_input("ピーク同値許容（ATR倍）", value=0.55, step=0.10, min_value=0.0)
dbl_tol_pct  = st.sidebar.number_input("ピーク同値許容（％）", value=0.10, step=0.05, min_value=0.0)
dbl_min_sep  = st.sidebar.slider("2点の最低バー間隔", 3, 80, 10, 1)
dbl_min_depth_atr = st.sidebar.number_input("谷/山の最小深さ（ATR倍）", value=0.90, step=0.10, min_value=0.0)
dbl_require_confirm = st.sidebar.checkbox("ネックライン確定（終値でブレイク）を必須にする", value=False)
show_only_latest_double = st.sidebar.checkbox("ダブルトップ/ダブルボトムは直近のみ表示", value=False)
flag_lookback = st.sidebar.slider("フラッグ/ペナント", 20, 600, 100, 5)
hs_lookback = st.sidebar.slider("ヘッド＆ショルダーズ（逆含む）", 20, 600, 175, 5)
pat_min_touches = st.sidebar.slider("最小接触回数（線へのタッチ数）", 2, 5, 2)
pat_tol_price = st.sidebar.number_input("価格許容誤差（ダブル/ボックス）", value=0.10, step=0.05)

# Flag/Pennant パラメータ
st.sidebar.caption("— フラッグ/ペナント 設定 —")
flag_Npush = st.sidebar.slider("旗竿推定 Npush（本）", 10, 60, 30, 2)
flag_min_bars = st.sidebar.slider("調整ゾーン最小本数", 6, 30, 10, 2)
flag_max_bars = st.sidebar.slider("調整ゾーン最大本数", 10, 80, 40, 2)
flag_sigma_k = st.sidebar.slider("σバンド倍率（境界）", 0.5, 3.0, 1.0, 0.5)
flag_pole_min_atr = st.sidebar.slider("旗竿の最小強度（ATR倍）", 1.0, 5.0, 2.0, 0.5)

# H&S パラメータ
st.sidebar.caption("— ヘッド＆ショルダーズ 設定 —")
hs_tol = st.sidebar.slider("肩の高さ許容（比率）", 0.001, 0.02, 0.003, 0.001)

# ---- パターン検出実行 ----
patterns = []
try:
    double_patterns = []
    tri_df = None
    rect_df = None
    if enable_tri:
        tri_df = detect_triangles_df(
            df,
            high_col="high", low_col="low", close_col="close",
            atr_window=14,
            cons_min_bars=12, cons_max_bars=tri_lookback if 'tri_lookback' in locals() else 36,
            flat_tol_norm=0.0012,
            converge_min=0.25,
            width_max_atr=3.5,
            breakout_buffer_atr=0.30,
            confirm_bars=1,
            pretrend_win=24,
            require_breakout=False,
        )
        tri_df = tri_df[tri_df["quality_score"] >= 0.55].reset_index(drop=True)
        tri_df = tri_df.replace({None: np.nan})
    if enable_rect:
        rect_df = detect_rectangles_df(
            df,
            high_col="high", low_col="low", close_col="close",
            atr_window=14,
            win_min_bars=10,
            win_max_bars=rect_lookback if 'rect_lookback' in locals() else 36,
            pivot_lb=2,
            pivot_ub=2,
            flat_tol_norm=0.0009,
            drift_tol_norm=0.0007,
            width_max_atr=3.5,
            width_stability_max=0.28,
        )
        rect_df = rect_df[rect_df["quality_score"] >= 0.55].reset_index(drop=True)
        rect_df = rect_df.replace({None: np.nan})

        tol_atrK=0.55,
        min_sep_bars=10,
        min_depth_atr=0.90,
        confirm_bars=2,
        neck_break_atr=0.15,
        retest_within=30,
        retest_tol_atr=0.25,
        pretrend_win=24,
        pretrend_min=0.0,
    # ...existing code...
    patterns += [max(double_patterns, key=lambda p: p.t_end)] if (show_only_latest_double and double_patterns) else double_patterns


    if enable_flag:
        patterns_df = detect_flag_pennant(df)
        import pandas as pd
        if not isinstance(patterns_df, pd.DataFrame):
            try:
                patterns_df = pd.DataFrame(patterns_df)
            except Exception:
                patterns_df = pd.DataFrame()
        patterns += patterns_df.to_dict('records')
    if enable_hs:
        patterns += detect_head_shoulders(df, pivot_high, pivot_low, lookback=hs_lookback, tol=hs_tol)
except Exception as e:
    st.error(f"パターン検出中にエラー: {e}")
    patterns = []

# ---- すべてのチャートパターン表を結合して表示 ----
import pandas as pd
pattern_tables = []
if 'rect_df' in locals() and rect_df is not None and not rect_df.empty:
    pattern_tables.append(rect_df)
if 'tri_df' in locals() and tri_df is not None and not tri_df.empty:
    pattern_tables.append(tri_df)
if 'double_patterns' in locals() and double_patterns:
    double_df = pd.DataFrame(double_patterns)
    if not double_df.empty:
        pattern_tables.append(double_df)

# フラッグ/ペナント



import re
import pandas as pd

# ---- すべてのチャートパターン表を結合して表示 ----
pattern_tables = []
if 'rect_df' in locals() and rect_df is not None and not rect_df.empty:
    pattern_tables.append(rect_df)
if 'tri_df' in locals() and tri_df is not None and not tri_df.empty:
    pattern_tables.append(tri_df)
if 'double_patterns' in locals() and double_patterns:
    double_df = pd.DataFrame(double_patterns)
    if not double_df.empty:
        pattern_tables.append(double_df)

# patternsリスト（ヘッド＆ショルダーズ、フラッグ/ペナント等）も表に追加
if patterns:
    patterns_dicts = []
    for p in patterns:
        if isinstance(p, re.Pattern):
            d = {"pattern": p.pattern, "flags": p.flags}
        elif hasattr(p, "keys"):
            d = dict(p)
        elif hasattr(p, "__dict__"):
            d = vars(p)
        else:
            d = {"value": str(p)}
        mt = measured_targets(p)
        d['dn'] = mt.get('down', 0.0)
        patterns_dicts.append(d)
    patterns_df = pd.DataFrame(patterns_dicts)
    if not patterns_df.empty:
        pattern_tables.append(patterns_df)

if pattern_tables:
    all_patterns_df = pd.concat(pattern_tables, ignore_index=True, sort=False)
    st.dataframe(all_patterns_df.style.format({
        "width_mean":"{:.3f}", "width_std":"{:.3f}", "width_stability":"{:.2f}",
        "quality_score":"{:.2f}", "quality":"{:.2f}",
        "entry":"{:.3f}", "stop":"{:.3f}", "target":"{:.3f}",
        "dn":"{:.3f}",
    }, na_rep="-"))

# ---- すべてのチャートパターン表を結合して表示 ----
import pandas as pd
pattern_tables = []
if 'rect_df' in locals() and rect_df is not None and not rect_df.empty:
    pattern_tables.append(rect_df)
if 'tri_df' in locals() and tri_df is not None and not tri_df.empty:
    pattern_tables.append(tri_df)
if 'double_patterns' in locals() and double_patterns:
    double_df = pd.DataFrame(double_patterns)
    if not double_df.empty:
        pattern_tables.append(double_df)

if pattern_tables:
    all_patterns_df = pd.concat(pattern_tables, ignore_index=True, sort=False)
    st.dataframe(all_patterns_df.style.format({
        "width_mean":"{:.3f}", "width_std":"{:.3f}", "width_stability":"{:.2f}",
        "quality_score":"{:.2f}",
        "entry":"{:.3f}", "stop":"{:.3f}", "target":"{:.3f}",
    }, na_rep="-"))

# ---------------- モデル読み込み（TTL付きキャッシュ） ----------------
@st.cache_resource(show_spinner=False, ttl=600)
def _load_break_model_cached(path: str):
    return joblib.load(path)

def load_break_model(path: str):
    try:
        obj = _load_break_model_cached(path)
        return obj["model"], obj.get("Xcols", None), obj.get("meta", {})
    except Exception:
        return None, None, None

def session_onehot_feat(ts):
    h = ts.hour
    return (1.0 if 9<=h<15 else 0.0,
            1.0 if 16<=h<24 else 0.0,
            1.0 if h>=22 or h<5 else 0.0)

def make_features_for_level(df, ts, level, dir_sign, touch_buffer, trend_look=150):
    feat = {}
    features_list = [
        "ret_1", "ret_4", "atr", "touch_density", "slope_long", "tokyo", "london", "ny", "ret_8", "ret_12",
        "atr14_norm", "atr_ratio", "d_atr14", "range_pct", "body_ratio", "wick_up_ratio", "wick_dn_ratio", "slope_short_6",
        "z_close_20", "sin_hour", "cos_hour", "touch_x_atr", "slope_long_x_ny", "ret_1_v", "ret_4_v", "ret_8_v", "ret_12_v",
        "range_pct_v", "slope_short_6_v", "atr14_norm_v", "atr_ratio_56", "rv20", "reg_atr_low", "reg_atr_mid", "reg_atr_high",
        "tokyo_x_range", "tokyo_x_touch", "london_x_range", "london_x_touch", "ny_x_range", "ny_x_touch", "touch_x_atr_v",
        "touch_x_slope_v", "z_x_atr_v", "reg_low_x_z", "reg_low_x_slope_v", "reg_mid_x_z", "reg_mid_x_slope_v", "reg_high_x_z",
        "reg_high_x_slope_v", "z_close_20_sq", "slope_short_6_v_sq", "ret_8_v_sq", "ret_12_v_sq", "dir", "dist_to_level",
        "atr_slope_dir", "rsi_div_dir", "high_low_ratio_20", "atr_change_10"
    ]
    print(f"CALL make_features_for_level: df={df.shape}, ts={ts}, level={level}, dir_sign={dir_sign}")
    print(f"features_list={features_list}")
    filtered_feat = {k: feat.get(k, 0.0) for k in features_list}
    print(f"filtered_feat keys={list(filtered_feat.keys())}, len={len(filtered_feat)}")
    with open("filtered_feat_debug.txt", "a", encoding="utf-8") as dbg:
        dbg.write(f"CALL make_features_for_level: df={getattr(df, 'shape', None)}, ts={ts}, level={level}, dir_sign={dir_sign}\n")
    feat = {}
    i = len(df)-1
    c  = float(df["close"].iloc[i])
    h  = float(df["high"].iloc[i])
    l  = float(df["low"].iloc[i])
    feat["dir"] = dir_sign
    feat["dist_to_level"] = (c - level) * dir_sign
    feat["diff_level_close"] = (c - level) * dir_sign
    feat["recent_high"] = float(df["high"].iloc[max(0, i-10):i+1].max()) * dir_sign
    feat["recent_low"] = float(df["low"].iloc[max(0, i-10):i+1].min()) * dir_sign
    feat["above_level"] = 1.0 if c > level else 0.0
    feat["below_level"] = 1.0 if c < level else 0.0
    a = atr(df, 14).fillna(0.0).iloc[i]
    feat["atr_norm"] = a / max(1e-6, c)
    feat["atr_slope_dir"] = (atr(df, 14).diff().fillna(0.0).iloc[i]) * dir_sign
    if "rsi" in df.columns:
        feat["rsi_div_dir"] = (df["rsi"].diff().fillna(0.0).iloc[i]) * dir_sign
    else:
        feat["rsi_div_dir"] = 0.0
    dist = abs(c - level)
    feat["near"] = 1.0 / (dist + 1e-6)
    Ntouch=200
    sub = df.iloc[max(0, i-Ntouch):i+1]
    touches = int((((sub["Low"]<=level)&(sub["High"]>=level)) | (sub["Close"].sub(level).abs()<=touch_buffer)).sum()) if "Low" in df.columns else int((((sub["low"]<=level)&(sub["high"]>=level)) | (sub["close"].sub(level).abs()<=touch_buffer)).sum())
    feat["touches"] = touches
    sess_tokyo, sess_london, sess_ny = session_onehot_feat(ts)
    feat["tokyo"] = sess_tokyo
    feat["london"] = sess_london
    feat["ny"] = sess_ny
    # --- 不足特徴量の追加 ---
    # z_close_20_sq: z_close_20の2乗
    if "z_close_20" in feat:
        feat["z_close_20_sq"] = feat["z_close_20"] ** 2
    else:
        feat["z_close_20_sq"] = 0.0
    # slope_short_6_v_sq: slope_short_6_vの2乗
    if "slope_short_6_v" in feat:
        feat["slope_short_6_v_sq"] = feat["slope_short_6_v"] ** 2
    else:
        feat["slope_short_6_v_sq"] = 0.0
    # ret_8_v_sq: ret_8_vの2乗
    if "ret_8_v" in feat:
        feat["ret_8_v_sq"] = feat["ret_8_v"] ** 2
    else:
        feat["ret_8_v_sq"] = 0.0
    # ret_12_v_sq: ret_12_vの2乗
    if "ret_12_v" in feat:
        feat["ret_12_v_sq"] = feat["ret_12_v"] ** 2
    else:
        feat["ret_12_v_sq"] = 0.0
    # high_low_ratio_20: 直近20本のhigh/low比
    if i >= 19:
        high20 = float(df["high"].iloc[i-19:i+1].max())
        low20 = float(df["low"].iloc[i-19:i+1].min())
        feat["high_low_ratio_20"] = high20 / max(low20, 1e-6)
    else:
        feat["high_low_ratio_20"] = 0.0
    # atr_change_10: ATRの10本変化量
    atr_series = atr(df, 14).fillna(0.0)
    if i >= 10:
        feat["atr_change_10"] = atr_series.iloc[i] - atr_series.iloc[i-10]
    else:
        feat["atr_change_10"] = 0.0
    # --- 必須特徴量の追加（漏れ防止） ---
    for k in ["z_close_20_sq", "slope_short_6_v_sq", "ret_8_v_sq", "ret_12_v_sq", "dir", "dist_to_level", "atr_slope_dir", "rsi_div_dir", "high_low_ratio_20", "atr_change_10"]:
        if k not in feat:
            feat[k] = 0.0
    # rsi_div_dir
    if "rsi_div_dir" not in feat:
        feat["rsi_div_dir"] = 0.0
    # high_low_ratio_20
    if "high_low_ratio_20" not in feat:
        feat["high_low_ratio_20"] = 0.0
    # atr_change_10
    if "atr_change_10" not in feat:
        feat["atr_change_10"] = 0.0
    feat = {}
    # ...既存の特徴量生成処理...
    # ここに基本特徴量の代入が続く
    feat["dir"] = dir_sign
    feat["dist_to_level"] = (float(df["close"].iloc[len(df)-1]) - level) * dir_sign
    # ...（他の特徴量生成処理）...

    # --- 不足特徴量の追加 ---
    # z_close_20_sq: z_close_20の2乗
    if "z_close_20" in feat:
        feat["z_close_20_sq"] = feat["z_close_20"] ** 2
    else:
        feat["z_close_20_sq"] = 0.0

    # slope_short_6_v_sq: slope_short_6_vの2乗
    if "slope_short_6_v" in feat:
        feat["slope_short_6_v_sq"] = feat["slope_short_6_v"] ** 2
    else:
        feat["slope_short_6_v_sq"] = 0.0

    # ret_8_v_sq: ret_8_vの2乗
    if "ret_8_v" in feat:
        feat["ret_8_v_sq"] = feat["ret_8_v"] ** 2
    else:
        feat["ret_8_v_sq"] = 0.0

    # ret_12_v_sq: ret_12_vの2乗
    if "ret_12_v" in feat:
        feat["ret_12_v_sq"] = feat["ret_12_v"] ** 2
    else:
        feat["ret_12_v_sq"] = 0.0

    # high_low_ratio_20: 直近20本のhigh/low比
    i = len(df)-1
    if i >= 19:
        high20 = float(df["high"].iloc[i-19:i+1].max())
        low20 = float(df["low"].iloc[i-19:i+1].min())
        feat["high_low_ratio_20"] = high20 / max(low20, 1e-6)
    else:
        feat["high_low_ratio_20"] = 0.0

    # atr_change_10: ATRの10本変化量
    atr_series = atr(df, 14).fillna(0.0)
    if i >= 10:
        feat["atr_change_10"] = atr_series.iloc[i] - atr_series.iloc[i-10]
    else:
        feat["atr_change_10"] = 0.0
    features_list = [
        "ret_1", "ret_4", "atr", "touch_density", "slope_long", "tokyo", "london", "ny", "ret_8", "ret_12",
        "atr14_norm", "atr_ratio", "d_atr14", "range_pct", "body_ratio", "wick_up_ratio", "wick_dn_ratio", "slope_short_6",
        "z_close_20", "sin_hour", "cos_hour", "touch_x_atr", "slope_long_x_ny", "ret_1_v", "ret_4_v", "ret_8_v", "ret_12_v",
        "range_pct_v", "slope_short_6_v", "atr14_norm_v", "atr_ratio_56", "rv20", "reg_atr_low", "reg_atr_mid", "reg_atr_high",
        "tokyo_x_range", "tokyo_x_touch", "london_x_range", "london_x_touch", "ny_x_range", "ny_x_touch", "touch_x_atr_v",
        "touch_x_slope_v", "z_x_atr_v", "reg_low_x_z", "reg_low_x_slope_v", "reg_mid_x_z", "reg_mid_x_slope_v", "reg_high_x_z",
        "reg_high_x_slope_v", "z_close_20_sq", "slope_short_6_v_sq", "ret_8_v_sq", "ret_12_v_sq", "dir", "dist_to_level",
        "atr_slope_dir", "rsi_div_dir", "high_low_ratio_20", "atr_change_10"
    ]
    # 必要な特徴量のみ抽出し、0埋め（必ず60個のkeyを持つdictを返す）
    filtered_feat = {k: feat.get(k, 0.0) for k in features_list}
    debug_path = r"c:\Users\daiki\OneDrive\fx2\FX-Learning-Tools\filtered_feat_debug.txt"
    try:
        with open(debug_path, "a", encoding="utf-8") as dbg:
            dbg.write(f"CALL make_features_for_level: df={df.shape}, ts={ts}, level={level}, dir_sign={dir_sign}\n")
            dbg.write(f"features_list={features_list}\n")
            dbg.write(f"filtered_feat keys={list(filtered_feat.keys())}, len={len(filtered_feat)}\n")
    except Exception as e:
        with open(debug_path, "a", encoding="utf-8") as dbg:
            dbg.write(f"EXCEPTION in make_features_for_level: {e}\n")
    # レベルに最も近いバーを探す
    i = len(df)-1  # 直近バー
    # カラム名を学習時と揃える
    c  = float(df["close"].iloc[i])
    h  = float(df["high"].iloc[i])
    l  = float(df["low"].iloc[i])
    # 方向性特徴量
    feat = {}
    feat["dir"] = dir_sign
    feat["dist_to_level"] = (c - level) * dir_sign
    feat["diff_level_close"] = (c - level) * dir_sign
    feat["recent_high"] = float(df["high"].iloc[max(0, i-10):i+1].max()) * dir_sign
    feat["recent_low"] = float(df["low"].iloc[max(0, i-10):i+1].min()) * dir_sign
    feat["above_level"] = 1.0 if c > level else 0.0
    feat["below_level"] = 1.0 if c < level else 0.0
    a = atr(df, 14).fillna(0.0).iloc[i]
    feat["atr_norm"] = a / max(1e-6, c)
    feat["atr_slope_dir"] = (atr(df, 14).diff().fillna(0.0).iloc[i]) * dir_sign
    # RSIダイバージェンス例（仮）
    if "rsi" in df.columns:
        feat["rsi_div_dir"] = (df["rsi"].diff().fillna(0.0).iloc[i]) * dir_sign
    else:
        feat["rsi_div_dir"] = 0.0
    dist     = abs(c - level)
    feat["near"] = 1.0 / (dist + 1e-6)
    Ntouch=200
    sub = df.iloc[max(0, i-Ntouch):i+1]
    touches = int((((sub["Low"]<=level)&(sub["High"]>=level)) | (sub["Close"].sub(level).abs()<=touch_buffer)).sum()) if "Low" in df.columns else int((((sub["low"]<=level)&(sub["high"]>=level)) | (sub["close"].sub(level).abs()<=touch_buffer)).sum())
    feat["touches"] = touches
    sess_tokyo, sess_london, sess_ny = session_onehot_feat(ts)
    feat["tokyo"] = sess_tokyo
    feat["london"] = sess_london
    feat["ny"] = sess_ny
    # meta["features"]のみでフィルタ（余計なカラム除外）
    import json
    with open("models/break_meta.json", "r", encoding="utf-8") as f:
        meta_json = json.load(f)
        features_list = meta_json.get("features", list(feat.keys()))
    # 必要な特徴量のみ抽出し、0埋め（必ず60個のkeyを持つdictを返す）
    filtered_feat = {k: feat.get(k, 0.0) for k in features_list}
    filtered_feat["timestamp"] = ts
    print(f"filtered_feat keys={list(filtered_feat.keys())}, len={len(filtered_feat)}")
    return filtered_feat

    # テスト用: make_features_for_levelの直接呼び出し
    if __name__ == "__main__":
        import pandas as pd
        from datetime import datetime
        # ダミーデータ作成
        df = pd.DataFrame({
            "close": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "open": [100, 100, 101, 102, 103],
        }, index=pd.date_range("2025-09-01", periods=5, freq="D"))
        ts = df.index[-1]
        level = 102
        dir_sign = 1
        touch_buffer = 0.5
        make_features_for_level(df, ts, level, dir_sign, touch_buffer)

# ====================== チャート描画 ======================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="Price",
    increasing_line_color=COLOR_CANDLE_UP_EDGE, increasing_fillcolor=COLOR_CANDLE_UP_BODY,
    decreasing_line_color=COLOR_CANDLE_DN_EDGE, decreasing_fillcolor=COLOR_CANDLE_DN_BODY
))
## --- トレードポイント描画 ---
for pt in st.session_state.get("trade_points", []):
    if pt["type"] == "buy":
        fig.add_shape(type="line", x0=pt["time"], x1=pt["time"], y0=pt["price"]-0.1, y1=pt["price"]+0.1,
                      line=dict(color="blue", width=3))
        fig.add_annotation(x=pt["time"], y=pt["price"], text="Buy", showarrow=True, arrowhead=2, font=dict(color="blue"))
        fig.add_shape(type="line", x0=pt["time"], x1=pt["time"], y0=pt["tp"]-0.1, y1=pt["tp"]+0.1,
                      line=dict(color="green", width=2, dash="dot"))
        fig.add_annotation(x=pt["time"], y=pt["tp"], text="TP", showarrow=True, arrowhead=1, font=dict(color="green"))
        fig.add_shape(type="line", x0=pt["time"], x1=pt["time"], y0=pt["sl"]-0.1, y1=pt["sl"]+0.1,
                      line=dict(color="red", width=2, dash="dot"))
        fig.add_annotation(x=pt["time"], y=pt["sl"], text="SL", showarrow=True, arrowhead=1, font=dict(color="red"))
    elif pt["type"] == "sell":
        fig.add_shape(type="line", x0=pt["time"], x1=pt["time"], y0=pt["price"]-0.1, y1=pt["price"]+0.1,
                      line=dict(color="orange", width=3))
        fig.add_annotation(x=pt["time"], y=pt["price"], text="Sell", showarrow=True, arrowhead=2, font=dict(color="orange"))
        fig.add_shape(type="line", x0=pt["time"], x1=pt["time"], y0=pt["tp"]-0.1, y1=pt["tp"]+0.1,
                      line=dict(color="green", width=2, dash="dot"))
        fig.add_annotation(x=pt["time"], y=pt["tp"], text="TP", showarrow=True, arrowhead=1, font=dict(color="green"))
        fig.add_shape(type="line", x0=pt["time"], x1=pt["time"], y0=pt["sl"]-0.1, y1=pt["sl"]+0.1,
                      line=dict(color="red", width=2, dash="dot"))
        fig.add_annotation(x=pt["time"], y=pt["sl"], text="SL", showarrow=True, arrowhead=1, font=dict(color="red"))
    elif pt["type"] == "tp":
        fig.add_shape(type="line", x0=pt["time"], x1=pt["time"], y0=pt["price"]-0.1, y1=pt["price"]+0.1,
                      line=dict(color="green", width=2, dash="dot"))
        fig.add_annotation(x=pt["time"], y=pt["price"], text="TP", showarrow=True, arrowhead=1, font=dict(color="green"))
    elif pt["type"] == "sl":
        fig.add_shape(type="line", x0=pt["time"], x1=pt["time"], y0=pt["price"]-0.1, y1=pt["price"]+0.1,
                      line=dict(color="red", width=2, dash="dot"))
        fig.add_annotation(x=pt["time"], y=pt["price"], text="SL", showarrow=True, arrowhead=1, font=dict(color="red"))

# 水平線
score_df = compute_level_scores(df, levels, touch_buffer, w_touch, w_recent, w_session, w_vol)
if not score_df.empty:
    sc_min, sc_max = score_df["score"].min(), score_df["score"].max()
    rng = max(1e-9, (sc_max - sc_min))
    for _, r in score_df.iterrows():
        # スコアが全て同じ場合は固定値
        if math.isclose(sc_min, sc_max):
            alpha = 0.25
        else:
            alpha = 0.20 + 0.65 * (r["score"]-sc_min)/rng
        # alphaの範囲を保証
        alpha = min(max(alpha, 0), 1)
        fig.add_hline(
            y=r["level"],
            opacity=float(alpha),
            line_color=COLOR_LEVEL,
            line_width=1.5
        )
else:
    for lv in levels:
        fig.add_hline(
            y=lv,
            opacity=0.35,
            line_color=COLOR_LEVEL,
            line_width=1.2
        )

# トレンド＆チャネル
trend = regression_trend(df, reg_lookback, use="low")
if trend:
    fig.add_shape(type="line", x0=trend["x0"], y0=trend["y0"], x1=trend["x1"], y1=trend["y1"],
                  line=dict(width=2.4, color=COLOR_TREND))
    if trend["sigma"] > 0:
        x0, x1 = trend["x0"], trend["x1"]
        y0_u = trend["y0"] + chan_k*trend["sigma"]; y1_u = trend["y1"] + chan_k*trend["sigma"]
        y0_l = trend["y0"] - chan_k*trend["sigma"]; y1_l = trend["y1"] - chan_k*trend["sigma"]
        fig.add_shape(type="line", x0=x0, y0=y0_u, x1=x1, y1=y1_u,
                      line=dict(width=1.6, color=COLOR_CH_UP, dash="dot"))
        fig.add_shape(type="line", x0=x0, y0=y0_l, x1=x1, y1=y1_l,
                      line=dict(width=1.6, color=COLOR_CH_DN, dash="dot"))

# パターン描画

def _line_points(i1, i2, slope, intercept):
    xs = np.array([i1, i2], dtype=float)
    ys = slope * xs + intercept
    return xs.astype(int), ys

if 'tri_df' in locals() and tri_df is not None and not tri_df.empty:
    tri_df_sorted = tri_df.sort_values("quality_score", ascending=False).head(10)
    for _, r in tri_df_sorted.iterrows():
        s, e = int(r["start_idx"]), int(r["end_idx"])
        xs_u, ys_u = _line_points(s, e, r["upper_slope"], r["upper_intercept"])
        xs_l, ys_l = _line_points(s, e, r["lower_slope"], r["lower_intercept"])
        x_u = df.index[xs_u] if "time" not in df.columns else df["time"].iloc[xs_u]
        x_l = df.index[xs_l] if "time" not in df.columns else df["time"].iloc[xs_l]
        fig.add_scatter(x=x_u, y=ys_u, mode="lines", name=f"{r['type']} upper (q={r['quality_score']:.2f})", opacity=0.7)
        fig.add_scatter(x=x_l, y=ys_l, mode="lines", name=f"{r['type']} lower", opacity=0.7)
        fig.add_hline(y=r["entry"],  line_dash="dot", annotation_text="entry")
        fig.add_hline(y=r["stop"],   line_dash="dot", annotation_text="stop")
        fig.add_hline(y=r["target"], line_dash="dot", annotation_text="target")

if 'rect_df' in locals():
    def _line_points(i1, i2, slope, intercept):
        xs = np.array([i1, i2], dtype=float)
        ys = slope * xs + intercept
        return xs.astype(int), ys

    if rect_df is not None:
        rect_df_sorted = rect_df.sort_values("quality_score", ascending=False).head(10)
        for _, r in rect_df_sorted.iterrows():
            s, e = int(r["start_idx"]), int(r["end_idx"])
            xs_u, ys_u = _line_points(s, e, r["upper_slope"], r["upper_intercept"])
            xs_l, ys_l = _line_points(s, e, r["lower_slope"], r["lower_intercept"])

            x_u = df.index[xs_u] if "time" not in df.columns else df["time"].iloc[xs_u]
            x_l = df.index[xs_l] if "time" not in df.columns else df["time"].iloc[xs_l]

            fig.add_scatter(x=x_u, y=ys_u, mode="lines", name=f"rect upper (q={r['quality_score']:.2f})", opacity=0.7)
            fig.add_scatter(x=x_l, y=ys_l, mode="lines", name="rect lower", opacity=0.7)

            fig.add_hline(y=r["entry"],  line_dash="dot", annotation_text="entry")
            fig.add_hline(y=r["stop"],   line_dash="dot", annotation_text="stop")
            fig.add_hline(y=r["target"], line_dash="dot", annotation_text="target")

def _draw_rectangle(fig, p: Pattern):
    params = p['params'] if isinstance(p, dict) else getattr(p, 'params', {})
    sub = df.loc[params["sub_start"] : params["sub_end"]]
    up = params["upper"]; dn = params["lower"]
    fig.add_hrect(y0=dn, y1=up, x0=sub.index[0], x1=sub.index[-1],
                  line_width=0, fillcolor="rgba(67,160,71,0.12)")
    fig.add_hline(y=up, line=dict(color=COLOR_RECTANGLE, width=2, dash="dot"))
    fig.add_hline(y=dn, line=dict(color=COLOR_RECTANGLE, width=2, dash="dot"))

def _draw_double(fig, p: Pattern):
    params = p['params'] if isinstance(p, dict) else getattr(p, 'params', {})
    if (p.get('kind') if isinstance(p, dict) else getattr(p, 'kind', None))=="double_top":
        top = params["top"]; neck = params["neck"]
        x1 = p.t_start; x2 = p.t_end
        # 2つの高値を線で結ぶ
        fig.add_trace(go.Scatter(x=[x1, x2], y=[top, top], mode="lines+markers+text",
                                 line=dict(color=COLOR_DOUBLE_TOP, width=2),
                                 marker=dict(color=COLOR_DOUBLE_TOP, size=10),
                                 text=["Top1", "Top2"], textposition="top center", showlegend=False))
        # ネックライン（ピンク点線）
        fig.add_trace(go.Scatter(x=[x1, x2], y=[neck, neck], mode="lines",
                                 line=dict(color=COLOR_DOUBLE_TOP, width=2, dash="dot"), showlegend=False))
        # ネックライン注釈
        fig.add_annotation(x=x1, y=neck, text="NeckL", showarrow=True, arrowhead=1, ax=0, ay=30, font=dict(color=COLOR_DOUBLE_TOP))
    elif (p.get('kind') if isinstance(p, dict) else getattr(p, 'kind', None))=="double_bottom":
        bot = params["bottom"]; neck = params["neck"]
        x1 = p.t_start; x2 = p.t_end
        # 2つの安値を線で結ぶ
        fig.add_trace(go.Scatter(x=[x1, x2], y=[bot, bot], mode="lines+markers+text",
                                 line=dict(color=COLOR_DOUBLE_BOTTOM, width=2),
                                 marker=dict(color=COLOR_DOUBLE_BOTTOM, size=10),
                                 text=["Bottom1", "Bottom2"], textposition="bottom center", showlegend=False))
        # ネックライン（青点線）
        fig.add_trace(go.Scatter(x=[x1, x2], y=[neck, neck], mode="lines",
                                 line=dict(color=COLOR_DOUBLE_BOTTOM, width=2, dash="dot"), showlegend=False))
        # ネックライン注釈
        fig.add_annotation(x=x1, y=neck, text="NeckL", showarrow=True, arrowhead=1, ax=0, ay=-30, font=dict(color=COLOR_DOUBLE_BOTTOM))

def _draw_flag(fig, p: Pattern):
    params = p['params'] if isinstance(p, dict) else getattr(p, 'params', {})
    sub = df.loc[params["sub_start"] : params["sub_end"]]
    x = np.arange(len(sub))
    m = params["slope"]; b = params["intercept"]; s = params["sigma"]; k = params["band_k"]
    y_mid = m*x + b
    y_up = y_mid + k*s
    y_dn = y_mid - k*s
    fig.add_trace(go.Scatter(x=sub.index, y=y_up, mode="lines", name="flag_up",
                             line=dict(color=COLOR_FLAG, width=2, dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=sub.index, y=y_dn, mode="lines", name="flag_dn",
                             line=dict(color=COLOR_FLAG, width=2, dash="dot"), showlegend=False))

def _draw_hs(fig, p: Pattern):
    params = p['params'] if isinstance(p, dict) else getattr(p, 'params', {})
    neck = params.get("neck", None)
    if neck is not None:
        fig.add_hline(y=float(neck), line=dict(color=COLOR_HS, width=2, dash="dash"))
    # 左肩・頭・右肩・ネック両端の位置にマーカーとラベルを追加
    left_shoulder_x = params.get("left_shoulder_x")
    left_shoulder_y = params.get("left_shoulder_y")
    head_x = params.get("head_x")
    head_y = params.get("head_y")
    right_shoulder_x = params.get("right_shoulder_x")
    right_shoulder_y = params.get("right_shoulder_y")
    neck_left_x = params.get("neck_left_x")
    neck_left_y = params.get("neck_left_y")
    neck_right_x = params.get("neck_right_x")
    neck_right_y = params.get("neck_right_y")
    # マーカー描画（値が存在する場合のみ）
    if left_shoulder_x is not None and left_shoulder_y is not None:
        fig.add_trace(go.Scatter(x=[left_shoulder_x], y=[left_shoulder_y], mode="markers+text",
            marker=dict(color=COLOR_HS, size=12, symbol="circle"),
            text=["左肩"], textposition="top center", showlegend=False))
    if head_x is not None and head_y is not None:
        fig.add_trace(go.Scatter(x=[head_x], y=[head_y], mode="markers+text",
            marker=dict(color=COLOR_HS, size=14, symbol="diamond"),
            text=["頭"], textposition="top center", showlegend=False))
    if right_shoulder_x is not None and right_shoulder_y is not None:
        fig.add_trace(go.Scatter(x=[right_shoulder_x], y=[right_shoulder_y], mode="markers+text",
            marker=dict(color=COLOR_HS, size=12, symbol="circle"),
            text=["右肩"], textposition="top center", showlegend=False))
    # ネック両端
    if neck_left_x is not None and neck_left_y is not None:
        fig.add_trace(go.Scatter(x=[neck_left_x], y=[neck_left_y], mode="markers+text",
            marker=dict(color=COLOR_HS, size=10, symbol="triangle-up"),
            text=["ネック左端"], textposition="bottom center", showlegend=False))
    if neck_right_x is not None and neck_right_y is not None:
        fig.add_trace(go.Scatter(x=[neck_right_x], y=[neck_right_y], mode="markers+text",
            marker=dict(color=COLOR_HS, size=10, symbol="triangle-up"),
            text=["ネック右端"], textposition="bottom center", showlegend=False))
    # パターン名注釈
    fig.add_annotation(x=(p.get('t_end') if isinstance(p, dict) else getattr(p, 't_end', None)), y=params.get("head", float(df['close'].iloc[-1])),
                       text=("H&S" if (p.get('kind') if isinstance(p, dict) else getattr(p, 'kind', None))=="head_shoulders" else "Inv H&S"),
                       showarrow=False, font=dict(color=COLOR_HS))

patterns_sorted = sorted(patterns, key=lambda p: getattr(p, 'quality', 0), reverse=True)[:10]
for p in patterns_sorted:
    kind = p.get('kind') if isinstance(p, dict) else getattr(p, 'kind', None)
    # トライアングルはtri_dfで描画済みなのでここでは描画しない
    if kind=="rectangle":
        _draw_rectangle(fig, p)
    elif kind in ("double_top","double_bottom"):
        _draw_double(fig, p)
    elif kind in ("flag_up","flag_dn","pennant"):
        _draw_flag(fig, p)
    elif kind in ("head_shoulders","inverse_head_shoulders"):
        _draw_hs(fig, p)

# ---- 赤影（重要度別ウィンドウ）を重ねる ----
if use_news_shade and not windows_df.empty:
    fig = add_news_shading_to_fig(fig, windows_df)

# --- 疑似チャート投影（ゴースト）コントロール ---
st.sidebar.markdown("---")
st.sidebar.subheader("疑似チャート投影（ゴースト）")
enable_ghost = st.sidebar.checkbox("パターンから将来パスを薄く重ねる", value=False)
ghost_h = st.sidebar.slider("投影本数（バー）", 10, 120, 40, help="何本先まで薄く描くか")
ghost_mode = st.sidebar.selectbox("方法", ["EV直線", "ボラ扇形(平均)"], index=0)
ghost_alpha = st.sidebar.slider("透明度", 0.05, 0.6, 0.18)
ghost_fan_k = st.sidebar.slider("扇形の幅k（σ係数）", 0.5, 3.0, 1.5, 0.5)
ghost_sims = st.sidebar.slider("ランダムウォーク本数（任意）", 0, 300, 0)

# --- ゴーストのための補助関数 ---
def _pattern_levels_for_prob(df, p: Pattern):
    upper_level = None; lower_level = None
    kind = p.get('kind') if isinstance(p, dict) else getattr(p, 'kind', None)
    params = p['params'] if isinstance(p, dict) else getattr(p, 'params', {})
    if kind and kind.startswith("triangle"):
        subp = df.loc[params["sub_start"]:params["sub_end"]]
        m1,b1 = params["upper"]; m2,b2 = params["lower"]
        x_now  = len(subp)-1
        y_u = float(m1*x_now + b1); y_l = float(m2*x_now + b2)
        upper_level, lower_level = y_u, y_l
    elif kind=="rectangle":
        upper_level, lower_level = float(params["upper"]), float(params["lower"])
    elif kind=="double_top":
        upper_level, lower_level = None, float(params["neck"])
    elif kind=="double_bottom":
        upper_level, lower_level = float(params["neck"]), None
    elif kind in ("flag_up","flag_dn","pennant"):
        upper_level, lower_level = float(params["upper_now"]), float(params["lower_now"])
    elif kind=="head_shoulders":
        upper_level, lower_level = None, float(params["neck"])
    elif kind=="inverse_head_shoulders":
        upper_level, lower_level = float(params["neck"]), None

    return upper_level, lower_level

def _pick_best_pattern(patterns: list[Pattern]) -> Pattern | None:
    if not patterns: return None
    pats = sorted(patterns, key=lambda x: (
        x.get('quality') if isinstance(x, dict) else getattr(x, 'quality', None),
        x.get('t_end') if isinstance(x, dict) else getattr(x, 't_end', None)
    ), reverse=True)
    return pats[0]

def _ghost_path_ev(df, P_up, P_dn, tgt_up_px, tgt_dn_px, h: int):
    w_up = float(P_up) if not np.isnan(P_up) else 0.5
    w_dn = float(P_dn) if not np.isnan(P_dn) else 0.5
    if (w_up + w_dn) <= 1e-9:
        w_up = w_dn = 0.5
    else:
        s = w_up + w_dn
        w_up /= s; w_dn /= s
    c0 = float(df["close"].iloc[-1])
    xs = np.arange(h+1)
    up_line = c0 + (tgt_up_px - c0) * (xs / h)
    dn_line = c0 + (tgt_dn_px - c0) * (xs / h)
    y = w_up*up_line + w_dn*dn_line
    return xs, y

def _ghost_cone_from_atr(df, h: int, k: float):
    c0 = float(df["close"].iloc[-1])
    _atr = atr(df, 14).iloc[-1] if len(df)>=14 else (df["high"].iloc[-1]-df["low"].iloc[-1])
    xs = np.arange(h+1)
    sigma = float(_atr)
    spread = k * sigma * np.sqrt(np.maximum(0, xs))
    mid = np.full_like(xs, c0, dtype=float)
    return xs, mid, mid+spread, mid-spread

def _rand_walks(df, h: int, n: int, drift_line: np.ndarray | None):
    if n <= 0: return []
    c0 = float(df["close"].iloc[-1])
    _atr = atr(df, 14).iloc[-1] if len(df)>=14 else (df["high"].iloc[-1]-df["low"].iloc[-1])
    sigma = float(_atr) * 0.6
    paths=[]
    for _ in range(n):
        steps = np.random.normal(loc=0.0, scale=sigma, size=h)
        y = np.r_[c0, c0 + np.cumsum(steps)]
        if drift_line is not None:
            y = 0.7*y + 0.3*drift_line
        paths.append(y)
    return paths

# === ゴースト投影（疑似チャート） ===
if enable_ghost:
    try:
        pat = _pick_best_pattern(patterns)
        c0 = float(df["close"].iloc[-1])
        tgt_up_px = c0; tgt_dn_px = c0
        P_up = P_dn = np.nan

        if pat is not None:
            meas = measured_targets(pat)
            tgt_up_px = c0 + float(meas.get("up", 0.0))
            tgt_dn_px = c0 + float(meas.get("down", 0.0))

            model, use_cols, meta = _load_model_and_meta()
            upper_level, lower_level = _pattern_levels_for_prob(df, pat)
            lower_level = lower_level if lower_level is not None else 0.0
            ts_now = df.index[-1]
            try:
                in_news = bool(is_in_any_window(pd.Series([ts_now]), windows_df[["start","end"]]).iloc[0])
            except Exception:
                # フォールバック: 重要度/±分方式の関数がある場合はそちらでも可
                in_news = False
            if upper_level is not None:
                feat_up = make_features_for_level(df, ts_now, upper_level, +1, touch_buffer)
                if isinstance(feat_up, dict):
                    row_up = {k: feat_up.get(k, 0.0) for k in use_cols}
                elif isinstance(feat_up, list):
                    row_up = dict(zip(use_cols, feat_up))
                else:
                    raise TypeError("feat_up must be dict or list")
                row_up["timestamp"] = ts_now
                df_up = pd.DataFrame([row_up])
                for col in use_cols:
                    if col not in df_up.columns:
                        df_up[col] = 0.0
                df_up = df_up[use_cols]
                df_up["timestamp"] = ts_now
                # A. 入力確認
                print("df_up input:", df_up[use_cols].to_dict(orient="records")[0])
                # B. predict_proba shape確認
                proba_up = model.predict_proba(df_up[use_cols])
                print("predict_proba(df_up) shape:", proba_up.shape)
                print("predict_proba(df_up):", proba_up)
                pred_up = predict_with_session_theta(df_up, model, use_cols, meta)
                P_up = float(pred_up["proba"].iloc[0])
                theta_up = float(pred_up["theta"].iloc[0])
                sess_up = str(pred_up["session"].iloc[0])
            if lower_level is not None:
                feat_dn = make_features_for_level(df, ts_now, lower_level, -1, touch_buffer)
                if isinstance(feat_dn, dict):
                    row_dn = {k: feat_dn.get(k, 0.0) for k in use_cols}
                elif isinstance(feat_dn, list):
                    row_dn = dict(zip(use_cols, feat_dn))
                else:
                    raise TypeError("feat_dn must be dict or list")
                row_dn["timestamp"] = ts_now
                df_dn = pd.DataFrame([row_dn])
                for col in use_cols:
                    if col not in df_dn.columns:
                        df_dn[col] = 0.0
                df_dn = df_dn[use_cols]
                df_dn["timestamp"] = ts_now
                # A. 入力確認
                print("df_dn input:", df_dn[use_cols].to_dict(orient="records")[0])
                # B. predict_proba shape確認
                proba_dn = model.predict_proba(df_dn[use_cols])
                print("predict_proba(df_dn) shape:", proba_dn.shape)
                print("predict_proba(df_dn):", proba_dn)
                pred_dn = predict_with_session_theta(df_dn, model, use_cols, meta)
                P_dn = float(pred_dn["proba"].iloc[0])
                theta_dn = float(pred_dn["theta"].iloc[0])
                sess_dn = str(pred_dn["session"].iloc[0])

        idx0 = df.index[-1]
        inferred = pd.infer_freq(df.index)
        freq = inferred if inferred else "T"
        future_idx = pd.date_range(idx0, periods=ghost_h+1, freq=freq, tz=idx0.tz)

        # Noneやnanの場合は0.0で埋める
        P_dn = P_dn if P_dn is not None and not np.isnan(P_dn) else 0.0
        EV_dn = EV_dn if 'EV_dn' in locals() and EV_dn is not None and not np.isnan(EV_dn) else 0.0
        if ghost_mode == "EV直線":
            xs, y = _ghost_path_ev(df, P_up, P_dn, tgt_up_px, tgt_dn_px, ghost_h)
            fig.add_trace(go.Scatter(
                x=future_idx, y=y, mode="lines",
                line=dict(color="rgba(255,255,255,0.9)", width=2, dash="dot"),
                name="ghost(EV)", showlegend=False, opacity=ghost_alpha
            ))

        else:  # ボラ扇形(平均)
            xs, mid, up, dn = _ghost_cone_from_atr(df, ghost_h, ghost_fan_k)
            xs_ev, y_ev = _ghost_path_ev(df, P_up, P_dn, tgt_up_px, tgt_dn_px, ghost_h)
            fig.add_trace(go.Scatter(x=future_idx, y=y_ev, mode="lines",
                                     line=dict(color="rgba(255,255,255,0.95)", width=2, dash="dot"),
                                     name="ghost(drift)", showlegend=False, opacity=ghost_alpha))
            x_poly = list(future_idx) + list(future_idx[::-1])
            y_poly = list(up) + list(dn[::-1])
            fig.add_trace(go.Scatter(
                x=x_poly, y=y_poly, fill="toself",
                line=dict(width=0), fillcolor=f"rgba(255,255,255,{ghost_alpha*0.35:.3f})",
                name="ghost(cone)", showlegend=False
            ))
            if ghost_sims and ghost_sims > 0:
                walks = _rand_walks(df, ghost_h, ghost_sims, y_ev)
                for w in walks:
                    fig.add_trace(go.Scatter(
                        x=future_idx, y=w, mode="lines",
                        line=dict(width=1, color="rgba(200,200,200,0.55)"),
                        showlegend=False, opacity=ghost_alpha*0.6
                    ))
    except Exception as e:
        st.warning(f"ゴースト投影で問題が発生しました: {e}")

# === チャートレイアウト & 表示 ===
fig.update_layout(template="plotly_dark", paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
                  title=f"{symbol} {interval} - Auto Lines (Dark)", xaxis_rangeslider_visible=False,
                  font=dict(color=COLOR_TEXT, size=12))
fig.update_xaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, showline=True, linecolor=COLOR_GRID)
fig.update_yaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, showline=True, linecolor=COLOR_GRID)

# ★ ここでズーム保持を追加
fig.update_layout(uirevision="fx-live")

# 描画
st.plotly_chart(fig, use_container_width=True)
st.caption(f"最終更新: {pd.Timestamp.now(tz=JST).strftime('%Y-%m-%d %H:%M:%S %Z')}")

# ---------------- 近傍ニュース判定（モード別） ----------------
def near_news(ts: pd.Timestamp) -> bool:
    if news_filter_mode == "重要度別（赤影と同じ）":
        return is_suppressed(ts, windows_df)
    else:
        if news_df.empty: return False
        win = pd.Timedelta(minutes=news_win)
        cond = (news_df["importance"] >= news_imp_min) & (news_df["time"].between(ts-win, ts+win))
        return bool(cond.any())

# ---------------- シグナル（直近バー） ----------------
st.subheader("📣 シグナル（直近バー）")
i_last = len(df)-1
c_last = float(df["close"].iloc[i_last]); h_last = float(df["high"].iloc[i_last]); l_last = float(df["low"].iloc[i_last])
ts_last = df.index[i_last]

# ソフト抑制：窓内ならブレイクバッファ/K を強化
if use_soft_suppress and near_news(ts_last):
    break_buffer = break_buffer_base + soft_break_add
    retest_wait_k = retest_wait_k_base + soft_K_add
else:
    break_buffer = break_buffer_base
    retest_wait_k = retest_wait_k_base

alerts = []
if signal_mode == "水平線ブレイク(終値)":
    for lv in levels:
        if (c_last > lv + break_buffer) and (l_last <= lv): alerts.append(("上抜け", f"Lv {lv:.3f}"))
        if (c_last < lv - break_buffer) and (h_last >= lv): alerts.append(("下抜け", f"Lv {lv:.3f}"))
elif signal_mode == "トレンドラインブレイク(終値)":
    if trend:
        tl_val = trend["y1"]
        if (c_last > tl_val + break_buffer) and (l_last <= tl_val): alerts.append(("トレンド上抜け", f"TL {tl_val:.3f}"))
        if (c_last < tl_val - break_buffer) and (h_last >= tl_val): alerts.append(("トレンド下抜け", f"TL {tl_val:.3f}"))
elif signal_mode == "チャネル上抜け/下抜け(終値)":
    if trend and trend["sigma"] > 0:
        up = trend["y1"] + chan_k*trend["sigma"]
        dn = trend["y1"] - chan_k*trend["sigma"]
        if c_last > up + break_buffer: alerts.append(("チャネル上抜け", f"UP {up:.3f}"))
        if c_last < dn - break_buffer: alerts.append(("チャネル下抜け", f"DN {dn:.3f}"))
elif signal_mode == "リテスト指値(水平線)":
    if i_last >= 1:
        c_prev = float(df["close"].iloc[i_last-1]); l_prev = float(df["low"].iloc[i_last-1]); h_prev = float(df["high"].iloc[i_last-1])
        for lv in levels:
            up_break_prev = (c_prev > lv + break_buffer) and (l_prev <= lv)
            dn_break_prev = (c_prev < lv - break_buffer) and (h_prev >= lv)
            if up_break_prev and abs(c_last - lv) <= touch_buffer: alerts.append(("リテスト買い候補", f"Lv {lv:.3f}"))
            if dn_break_prev and abs(c_last - lv) <= touch_buffer: alerts.append(("リテスト売り候補", f"Lv {lv:.3f}"))


# --- 直近10本のシグナル数で新規制御 ---
recent_signals = alerts[-10:]  # 直近10本
allow = True
if len(recent_signals) >= 4:   # 4件以上なら新規は出さない
    allow = False

if not alerts:
    st.info("直近バーではシグナルなし。")
else:
    for kind, msg in alerts:
        if apply_news_filter and near_news(ts_last):
            st.warning(f"抑制（ニュース近傍）: {kind} - {msg} @ {ts_last}")
        else:
            st.success(f"{kind}: {msg} @ {ts_last}")

# ---------------- バックテスト（Retest指数つき / ハード抑制は従来通り） ----------------
def compute_retest(series_close: pd.Series, target: float, start_idx: int, K: int, tol_abs: float):
    hits = 0; checks = 0; hit_once = False
    last = len(series_close) - 1
    for j in range(start_idx+1, min(start_idx+K+1, last+1)):
        px = float(series_close.iloc[j])
        checks += 1
        if abs(px - target) <= tol_abs:
            hits += 1
            hit_once = True
    idx_val = (hits / checks) if checks > 0 else 0.0
    return idx_val, hit_once

def backtest(df: pd.DataFrame, levels: list, fwd_n: int, break_buffer_arg: float,
             spread_pips: float, news_df: pd.DataFrame, news_win: int,
             news_imp_min: int, apply_news: bool, signal_mode: str, retest_wait_k_arg: int,
             touch_buffer: float):
    rows=[]
    if len(df) <= fwd_n+1:
        return pd.DataFrame(columns=["time","mode","level_or_val","dir","entry","exit","ret_pips","retest_index","retest_hit"])
    pv = pip_value("USDJPY")
    close_s = df["close"]
    use_imp_mode = (news_filter_mode == "重要度別（赤影と同じ）")
    for i in range(1, len(df)-fwd_n):
        t = df.index[i]
        c  = float(df["close"].iloc[i])
        l1 = float(df["low"].iloc[i-1]); h1 = float(df["high"].iloc[i-1])
        # ハード抑制
        if apply_news:
            if use_imp_mode:
                if is_suppressed(t, windows_df): 
                    continue
            else:
                if not news_df.empty:
                    win = pd.Timedelta(minutes=news_win)
                    if ((news_df["importance"] >= news_imp_min) & (news_df["time"].between(t-win, t+win))).any():
                        continue
        # 以降は従来通り
        if signal_mode == "水平線ブレイク(終値)":
            for lv in levels:
                if (c > lv + break_buffer_arg) and (l1 <= lv):
                    entry, exitp = c, float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, lv, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="水平ブレイク上", level_or_val=float(lv),
                                     dir="long", entry=entry, exit=exitp,
                                     ret_pips=(exitp-entry)/pv - spread_pips,
                                     retest_index=ri, retest_hit=rh))
                if (c < lv - break_buffer_arg) and (h1 >= lv):
                    entry, exitp = c, float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, lv, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="水平ブレイク下", level_or_val=float(lv),
                                     dir="short", entry=entry, exit=exitp,
                                     ret_pips=(entry-exitp)/pv - spread_pips,
                                     retest_index=ri, retest_hit=rh))
        elif signal_mode == "トレンドラインブレイク(終値)":
            if trend:
                tl = trend["y1"]
                if (c > tl + break_buffer_arg) and (l1 <= tl):
                    entry, exitp = c, float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, tl, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="TLブレイク上", level_or_val=float(tl),
                                     dir="long", entry=entry, exit=exitp,
                                     ret_pips=(exitp-entry)/pv - spread_pips,
                                     retest_index=ri, retest_hit=rh))
                if (c < tl - break_buffer_arg) and (h1 >= tl):
                    entry, exitp = c, float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, tl, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="TLブレイク下", level_or_val=float(tl),
                                     dir="short", entry=entry, exit=exitp,
                                     ret_pips=(entry-exitp)/pv - spread_pips,
                                     retest_index=ri, retest_hit=rh))

        elif signal_mode == "チャネル上抜け/下抜け(終値)":
            if trend and trend["sigma"] > 0:
                up = trend["y1"] + chan_k*trend["sigma"]
                dn = trend["y1"] - chan_k*trend["sigma"]
                if c > up + break_buffer_arg:
                    entry, exitp = c, float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, up, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="チャネル上抜け", level_or_val=float(up),
                                     dir="long", entry=entry, exit=exitp,
                                     ret_pips=(exitp-entry)/pv - spread_pips,
                                     retest_index=ri, retest_hit=rh))
                if c < dn - break_buffer_arg:
                    entry, exitp = c, float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, dn, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="チャネル下抜け", level_or_val=float(dn),
                                     dir="short", entry=entry, exit=exitp,
                                     ret_pips=(entry-exitp)/pv - spread_pips,
                                     retest_index=ri, retest_hit=rh))

        elif signal_mode == "リテスト指値(水平線)":
            K = int(retest_wait_k_arg)
            for lv in levels:
                up_break = (c > lv + break_buffer_arg) and (l1 <= lv)
                dn_break = (c < lv - break_buffer_arg) and (h1 >= lv)

                # 上方向ブレイク後、K 本以内にリテスト→その“リテスト時刻”で約定
                if up_break:
                    for j in range(i+1, min(i+K, len(df)-fwd_n)):
                        if abs(float(df["close"].iloc[j]) - lv) <= touch_buffer:
                            entry = float(df["close"].iloc[j]); exitp = float(df["close"].iloc[j+fwd_n])
                            ri, rh = compute_retest(close_s, lv, i, K, float(touch_buffer))
                            rows.append(dict(time=df.index[j], mode="リテスト(L)", level_or_val=float(lv),
                                             dir="long", entry=entry, exit=exitp,
                                             ret_pips=(exitp-entry)/pv - spread_pips,
                                             retest_index=ri, retest_hit=rh))
                            break

                # 下方向ブレイク後のリテスト
                if dn_break:
                    for j in range(i+1, min(i+K, len(df)-fwd_n)):
                        if abs(float(df["close"].iloc[j]) - lv) <= touch_buffer:
                            entry = float(df["close"].iloc[j]); exitp = float(df["close"].iloc[j+fwd_n])
                            ri, rh = compute_retest(close_s, lv, i, K, float(touch_buffer))
                            rows.append(dict(time=df.index[j], mode="リテスト(S)", level_or_val=float(lv),
                                             dir="short", entry=entry, exit=exitp,
                                             ret_pips=(entry-exitp)/pv - spread_pips,
                                             retest_index=ri, retest_hit=rh))
                            break

    return pd.DataFrame(rows)

bt_df = None
if run_bt:
    with st.spinner("バックテスト実行中..."):
        bt_df = backtest_rolling(
            df=df,
            fwd_n=fwd_n,
            break_buffer_arg=break_buffer,
            spread_pips=spread_pips,
            news_win=news_win,
            news_imp_min=news_imp_min,
            apply_news=apply_news_filter,
            signal_mode=signal_mode,
            retest_wait_k_arg=retest_wait_k,
            touch_buffer=touch_buffer
        )
    if bt_df is None or bt_df.empty:
        st.warning("トレードが生成されませんでした。パラメータ/モードを見直してください。")
    else:
        st.subheader("📊 バックテスト結果（" + signal_mode + "）")
        total = len(bt_df)
        wins = int((bt_df["ret_pips"] > 0).sum())
        mean = float(bt_df["ret_pips"].mean())
        std  = float(bt_df["ret_pips"].std()) if total > 1 else float("nan")
        sharpe = mean / std if std and not math.isnan(std) and std>1e-9 else float("nan")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("件数", total)
        c2.metric("勝率", f"{wins/total*100:.1f}%")
        c3.metric("平均pips", f"{mean:.2f}")
        c4.metric("Sharpe(擬似)", f"{sharpe:.2f}" if not math.isnan(sharpe) else "n/a")

        st.write("方向別：")
        st.dataframe(bt_df.groupby("dir")["ret_pips"].agg(["count","mean","std"]))

        st.write("シグナル別：")
        st.dataframe(bt_df.groupby("mode")["ret_pips"].agg(["count","mean","std"]).sort_values("mean", ascending=False))

        st.write("リテストあり/なし 比較：")
        comp = bt_df.copy()
        comp["retest_bucket"] = comp["retest_hit"].map({True:"あり", False:"なし"})
        st.dataframe(comp.groupby("retest_bucket")["ret_pips"].agg(
            件数="count", 勝率=lambda s: (s>0).mean()*100, 平均pips="mean", STD="std", Retest指数平均=lambda s: comp.loc[s.index,"retest_index"].mean()
        ).round({"勝率":1, "平均pips":2, "STD":2, "Retest指数平均":2}))

        ch = bt_df[bt_df["mode"].isin(["チャネル上抜け","チャネル下抜け"])]
        if not ch.empty:
            st.write("チャネル上抜け/下抜けの統計：")
            st.dataframe(ch.groupby("mode")["ret_pips"].agg(["count","mean","std"]).rename(
                index={"チャネル上抜け":"上抜け","チャネル下抜け":"下抜け"}).sort_values("mean", ascending=False))

# ---------------- 発注（任意・簡易） ----------------
st.sidebar.markdown("---")
st.subheader("🧪 発注（任意）")
side = st.selectbox("方向", ["BUY","SELL"], index=0)
order_units = st.sidebar.number_input("発注数量", value=1000, step=1000)
paper_trade = st.sidebar.checkbox("紙トレ（シミュレーション）を有効", value=True)
use_oanda = st.sidebar.checkbox("OANDA発注を有効", value=False)
oanda_token = st.sidebar.text_input("OANDA Token（env: OANDA_TOKEN）", value=os.getenv("OANDA_TOKEN",""))
oanda_account = st.sidebar.text_input("OANDA AccountID（env: OANDA_ACCOUNT）", value=os.getenv("OANDA_ACCOUNT",""))
oanda_env = st.sidebar.selectbox("OANDA環境", ["practice","live"], index=0)

if st.button("成行発注"):
    price_now = float(df["close"].iloc[-1]); logs = []
    if paper_trade:
        st.success(f"[紙トレ] {side} {order_units} @ {price_now:.3f}")
        logs.append({"type":"paper","side":side,"units":order_units,"price":price_now,"time":str(df.index[-1])})
    if use_oanda and oanda_token and oanda_account:
        try:
            import requests
            inst = "USD_JPY"
            base = "https://api-fxpractice.oanda.com" if oanda_env=="practice" else "https://api-fxtrade.oanda.com"
            headers = {"Authorization": f"Bearer {oanda_token}", "Content-Type":"application/json"}
            data = {"order":{"units": str(order_units if side=="BUY" else -order_units),
                             "instrument": inst, "timeInForce":"FOK","type":"MARKET","positionFill":"DEFAULT"}}
            r = requests.post(f"{base}/v3/accounts/{oanda_account}/orders", headers=headers, data=json.dumps(data), timeout=10)
            st.success(f"[OANDA] 成行 {side} 送信OK") if r.status_code in (200,201) else st.error(f"[OANDA] {r.status_code} {r.text}")
            logs.append({"type":"oanda","resp":r.json() if r.ok else r.text})
        except Exception as e:
            st.error(f"[OANDA] 例外: {e}")
        # if use_mt5:
        #     try:
        #         import MetaTrader5 as mt5
        #         if not mt5.initialize():
        #             st.error(f"[MT5] 初期化失敗: {mt5.last_error()}")
        #         else:
        #             symbol_mt5 = mt5_symbol
        #             mt5.symbol_select(symbol_mt5, True)
        #             lot = 0.1
        #             typ = mt5.ORDER_TYPE_BUY if side=="BUY" else mt5.ORDER_TYPE_SELL
        #             req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol_mt5, "volume": lot, "type": typ,
        #                    "deviation": 10, "magic": 123456, "comment": "AutoLines", "type_filling": mt5.ORDER_FILLING_FOK}
        #             res = mt5.order_send(req)
        #             st.success(f"[MT5] 成行 {side} OK: {res.order}") if res.retcode == mt5.TRADE_RETCODE_DONE else st.error(f"[MT5] 失敗: {res.retcode}")
        #             mt5.shutdown()
        #     except Exception as e:
        #         st.error(f"[MT5] 例外: {e}")
    st.session_state.setdefault("trade_logs", [])
    st.session_state["trade_logs"].extend(logs)

with st.expander("実行ログ"):
    st.write(st.session_state.get("trade_logs", []))

# ---------------- スコア表 ----------------
st.subheader("⭐ 重要度スコア（上位）")
if not score_df.empty:
    st.dataframe(score_df[["level","score","touches","session_ratio"]].head(15))
else:
    st.info("レベルが検出されていません。eps/min_samples/look を調整してください。")

# ---------------- 期待pips（水平線ブレイクの過去平均） ----------------

def _collect_trades_range_horizontal(df: pd.DataFrame,
                                     start_i: int,
                                     end_i: int,
                                     fwd_n: int,
                                     break_buffer: float,
                                     spread_pips: float,
                                     news_win: int,
                                     news_imp_min: int,
                                     apply_news: bool,
                                     retest_wait_k: int,
                                     touch_buffer: float) -> list:
    """backtest_rolling の for ループから、水平線ブレイク(終値) 部分だけを切り出した差分収集。
    start_i (inclusive) ～ end_i (exclusive) の i について追加トレード行を返す。
    """
    if end_i <= start_i:
        return []
    rows = []
    pv_local = pip_value("USDJPY")
    close_s = df["close"]
    for i in range(start_i, end_i):
        past = df.iloc[:i+1]
        t = past.index[-1]
        c  = float(past["close"].iloc[-1])
        l1 = float(past["low"].iloc[-2]); h1 = float(past["high"].iloc[-2])

        if apply_news:
            win_df_past = _get_news_windows_until_cached(t, news_imp_min)
            if _is_suppressed_at(t, win_df_past, news_win, news_imp_min, "水平線ブレイク(終値)"):
                continue

        # レベル抽出（過去のみ）
        try:
            piv_hi_past, piv_lo_past = swing_pivots(past, look)
            lvls_past = horizontal_levels(piv_hi_past, piv_lo_past, eps=eps, min_samples=min_samples)
        except Exception:
            lvls_past = []

        for lv in lvls_past:
            # 上方向
            if (c > lv + break_buffer) and (l1 <= lv):
                entry = c; exitp = float(df["close"].iloc[i+fwd_n])
                ri, rh = compute_retest(close_s, lv, i, int(retest_wait_k), float(touch_buffer))
                rows.append(dict(time=t, mode="水平ブレイク上", level_or_val=float(lv),
                                 dir="long", entry=entry, exit=exitp,
                                 ret_pips=(exitp-entry)/pv_local - spread_pips,
                                 retest_index=ri, retest_hit=rh))
            # 下方向
            if (c < lv - break_buffer) and (h1 >= lv):
                entry = c; exitp = float(df["close"].iloc[i+fwd_n])
                ri, rh = compute_retest(close_s, lv, i, int(retest_wait_k), float(touch_buffer))
                rows.append(dict(time=t, mode="水平ブレイク下", level_or_val=float(lv),
                                 dir="short", entry=entry, exit=exitp,
                                 ret_pips=(entry-exitp)/pv_local - spread_pips,
                                 retest_index=ri, retest_hit=rh))
    return rows


def compute_expected_pips_table_for_levels(df, levels, fwd_n, break_buffer, spread_pips,
                                           news_df, news_win, news_imp_min, apply_news_filter,
                                           touch_buffer, retest_wait_k) -> tuple:
    """差分集計版 期待pipsテーブル計算（水平線ブレイク）。

    キャッシュ構造 (st.session_state['ev_diff_cache']):
        (params_sig, last_i, prev_bar_len, agg_dict)
        agg_dict: {(level, dir): [sum_ret_pips, count]}

    増分ロジック:
        新しいバー到着で len(df) 増加時、評価対象 i 範囲 [last_i+1, len(df)-fwd_n) を差分計算。
        fwd_n の未来参照特性により end_i = len(df)-fwd_n が現在確定している最終 i。
    フォールバック条件:
        - キャッシュなし / パラメタ変更 / バー長後退 / ギャップ (bar 増加 > fwd_n*2) 等
        → フル backtest_rolling で再構築。
    """
    cache_key = "ev_diff_cache"
    # 署名拡張: レベル/回帰/チャネル関連パラメタも含める
    params_sig = (
        fwd_n, break_buffer, spread_pips,
        news_win, news_imp_min, int(apply_news_filter),
        touch_buffer, retest_wait_k,
        look, eps, min_samples, reg_lookback, chan_k
    )

    bar_len = len(df)
    if bar_len <= fwd_n + 2:
        return None, None

    state = st.session_state.get(cache_key)

    # min_start は backtest_rolling と同様に計算
    min_start = max(2, reg_lookback, look * 4)

    def _build_from_full():
        bt = backtest_rolling(
            df=df,
            fwd_n=fwd_n,
            break_buffer_arg=break_buffer,
            spread_pips=spread_pips,
            news_win=news_win,
            news_imp_min=news_imp_min,
            apply_news=apply_news_filter,
            signal_mode="水平線ブレイク(終値)",
            retest_wait_k_arg=retest_wait_k,
            touch_buffer=touch_buffer
        )
        if bt is None or bt.empty:
            st.session_state[cache_key] = (params_sig, min_start-1, bar_len, {})
            return None, None
        agg = {}
        for (lv, d), g in bt.groupby(["level_or_val", "dir"]):
            s = g["ret_pips"].sum(); c = len(g)
            # 新形式: [sum, count, last_hit_i]
            agg[(float(lv), d)] = [float(s), int(c), 0]
        last_i = bar_len - fwd_n - 1  # フル計算時点で確定している最大 i
        for k in agg.keys():
            agg[k][2] = last_i
        st.session_state[cache_key] = (params_sig, last_i, bar_len, agg)
        return _agg_to_frames(agg)

    def _agg_to_frames(agg_dict):
        if not agg_dict:
            return None, None
        rows_level = []
        for (lv, d), vals in agg_dict.items():
            if len(vals) == 2:
                s, c = vals
            else:
                s, c, _lh = vals
            if c > 0:
                rows_level.append(dict(level=lv, dir=d, avg=s/c, n=c))
        if not rows_level:
            return None, None
        import pandas as _pd
        by_level_dir = _pd.DataFrame(rows_level)
        by_dir = (by_level_dir.groupby("dir")
                                .apply(lambda g: _pd.Series(dict(avg=(g['avg']*g['n']).sum()/g['n'].sum(), n=g['n'].sum())))
                                .reset_index())
        # 整列（オプション）
        by_level_dir = by_level_dir.sort_values("avg", ascending=False)
        return by_level_dir, by_dir

    # フォールバック条件判定
    fallback = False
    if state is None:
        fallback = True
    else:
        prev_sig, last_i, prev_bar_len, agg = state
        if prev_sig != params_sig:
            fallback = True
        elif bar_len < prev_bar_len:  # データ巻き戻り
            fallback = True
        elif bar_len - prev_bar_len > fwd_n * 2:  # かなりギャップ（安全のため）
            fallback = True

    import time, hashlib
    stats = st.session_state.setdefault('ev_stats', {
        'fallback_count': 0,
        'diff_fail_count': 0,
        'last_mode': None,
        'last_elapsed_ms': None,
        'last_added_trades': 0,
        'agg_size': 0,
        'params_sig_hash': None,
    })

    if fallback:
        t0 = time.time()
        try:
            frames = _build_from_full()
            elapsed = (time.time() - t0) * 1000
            stats.update({
                'fallback_count': stats.get('fallback_count',0) + 1,
                'last_mode': 'full',
                'last_elapsed_ms': elapsed,
                'last_added_trades': None,
                'agg_size': len(st.session_state.get(cache_key,(None,None,None,{}))[3]) if st.session_state.get(cache_key) else 0,
                'params_sig_hash': hashlib.sha1(repr(params_sig).encode()).hexdigest()[:10]
            })
            return frames
        except Exception:
            stats['diff_fail_count'] += 1
            return None, None

    # 差分処理
    prev_sig, last_i, prev_bar_len, agg = state
    end_i_exclusive = bar_len - fwd_n
    start_i = max(min_start, last_i + 1)

    if start_i < end_i_exclusive:
        import time, hashlib
        t0 = time.time()
        try:
            new_rows = _collect_trades_range_horizontal(
                df=df,
                start_i=start_i,
                end_i=end_i_exclusive,
                fwd_n=fwd_n,
                break_buffer=break_buffer,
                spread_pips=spread_pips,
                news_win=news_win,
                news_imp_min=news_imp_min,
                apply_news=apply_news_filter,
                retest_wait_k=retest_wait_k,
                touch_buffer=touch_buffer,
            )
            add_cnt = 0
            for r in new_rows:
                key = (float(r["level_or_val"]), r["dir"]) 
                existing = agg.get(key)
                if existing is None:
                    agg[key] = [float(r["ret_pips"]), 1, end_i_exclusive-1]
                else:
                    if len(existing) == 2:
                        s, c = existing
                        agg[key] = [s + float(r["ret_pips"]), c + 1, end_i_exclusive-1]
                    else:
                        s, c, lh = existing
                        agg[key] = [s + float(r["ret_pips"]), c + 1, end_i_exclusive-1]
                add_cnt += 1
            last_i = end_i_exclusive - 1
            # TTL プルーニング
            ttl = int(st.session_state.get('level_ttl_bars', 800))
            pruned = 0
            if ttl > 0:
                to_del = []
                for k, vals in agg.items():
                    if len(vals) < 3:
                        continue
                    _s, _c, lh = vals
                    if last_i - lh > ttl:
                        to_del.append(k)
                for k in to_del:
                    del agg[k]
                pruned = len(to_del)
            st.session_state[cache_key] = (params_sig, last_i, bar_len, agg)
            elapsed = (time.time() - t0) * 1000
            stats.update({
                'last_mode': 'diff',
                'last_elapsed_ms': elapsed,
                'last_added_trades': add_cnt,
                'agg_size': len(agg),
                'params_sig_hash': hashlib.sha1(repr(params_sig).encode()).hexdigest()[:10],
                'pruned_last': pruned,
                'pruned_total': stats.get('pruned_total', 0) + pruned,
            })
        except Exception:
            stats['diff_fail_count'] += 1
            # 差分計算失敗時はフル再計算にフォールバック
            t0 = time.time()
            try:
                frames = _build_from_full()
                elapsed = (time.time() - t0) * 1000
                stats.update({
                    'last_mode': 'full(fallback)',
                    'last_elapsed_ms': elapsed,
                    'last_added_trades': None,
                    'agg_size': len(st.session_state.get(cache_key,(None,None,None,{}))[3]) if st.session_state.get(cache_key) else 0,
                    'params_sig_hash': hashlib.sha1(repr(params_sig).encode()).hexdigest()[:10]
                })
                return frames
            except Exception:
                return None, None
    # start_i >= end_i_exclusive の場合は新規 i なし → そのまま再利用
    return _agg_to_frames(agg)

# ---------------- 「今からのブレイク確率」＋ 期待値ランキング ----------------
if show_break_prob:
    st.subheader("🎯 今からの水平線ブレイク確率（次H本以内）")
    model, Xcols, meta = load_break_model("models/break_model.joblib")
    if model is None:
        st.info("学習モデル（models/break_model.joblib）が見つかりません。先に ai_train_break.py を実行してください。")
    else:
            ts_now = df.index[-1]
            # 重要度スコアで一位の水平線のみ表示
            if score_df is not None and not score_df.empty:
                top_level = float(score_df.sort_values("score", ascending=False)["level"].iloc[0])
                use_levels = [top_level]
            else:
                use_levels = [levels[0]] if levels else []
            try:
                # 水平線ブレイク確率テーブルの計算
                from build_level_break_prob_table import build_level_break_prob_table_cached
                prob_df = build_level_break_prob_table_cached(
                    df=df,
                    ts_now=None,  # ts_nowは使わないためNoneで明示
                    use_levels=use_levels,
                    use_cols=meta.get("features", use_cols),
                    touch_buffer=touch_buffer,
                    model=model,
                    meta=meta,
                    make_features_for_level=make_features_for_level,  # 特徴量生成関数
                    predict_with_session_theta=predict_with_session_theta,  # モデル推論関数
                    cache=True,
                )
                # Adaptive θ 用: 確率履歴更新と θ 再計算
                st.session_state.setdefault('prob_history', [])
                if prob_df is not None and not prob_df.empty:
                    for _, rr in prob_df.iterrows():
                        for pk in ["P_up", "P_dn"]:
                            pv = rr.get(pk, None)
                            if isinstance(pv, (int,float)) and not np.isnan(pv):
                                st.session_state['prob_history'].append((float(pv), pd.Timestamp.utcnow()))
                    # バッファ上限
                    st.session_state['prob_history'] = st.session_state['prob_history'][-3000:]
                # 価格履歴更新 (ATR 用) ※ close 列が存在すると仮定
                st.session_state.setdefault('recent_closes', [])
                if 'close' in df.columns:
                    st.session_state['recent_closes'].extend(list(df['close'].iloc[-5:]))  # 末尾を少量追記
                    st.session_state['recent_closes'] = st.session_state['recent_closes'][-5000:]
                # θ 更新
                adaptive_theta_val = update_adaptive_theta(st.session_state['prob_history'], window=300)
                st.session_state['theta_current'] = adaptive_theta_val if st.session_state.get('use_adaptive_theta') else st.session_state.get('theta_base', 0.60)
            except Exception as e:
                import traceback
                st.error(f"ブレイク確率テーブルの計算でエラー: {e}")
                st.error(traceback.format_exc())  # 詳細なエラー情報も表示
                prob_df = None

            ev_params = dict(
                fwd_n=fwd_n,
                break_buffer=break_buffer,
                spread_pips=spread_pips,
                news_df=news_df,
                news_win=news_win,
                news_imp_min=news_imp_min,
                apply_news_filter=apply_news_filter,
                touch_buffer=touch_buffer,
                retest_wait_k=retest_wait_k,
            )
            ev_table, ev_dir, ev_meta = get_ev_tables(
                df=df,
                levels=levels,
                params=ev_params,
                compute_fn=compute_expected_pips_table_for_levels,
                min_recompute_interval=3.0,
                max_bar_lookback_skip=1,
            )

            # EV 計算メタ情報表示
            with st.expander("EV計算メタ", expanded=False):
                stats = st.session_state.get('ev_stats', {})
                st.write({
                    'last_mode': stats.get('last_mode'),
                    'last_elapsed_ms': stats.get('last_elapsed_ms'),
                    'last_added_trades': stats.get('last_added_trades'),
                    'agg_size': stats.get('agg_size'),
                    'fallback_count': stats.get('fallback_count'),
                    'diff_fail_count': stats.get('diff_fail_count'),
                    'params_sig_hash': stats.get('params_sig_hash'),
                    'pruned_last': stats.get('pruned_last'),
                    'pruned_total': stats.get('pruned_total'),
                })
                # ニュースキャッシュ統計
                nstats = st.session_state.get('news_stats', {})
                if nstats:
                    st.write({'news_cache_hits': nstats.get('hits'), 'news_cache_miss': nstats.get('miss'), 'news_cache_size': nstats.get('size')})
                    if st.button("ニュースキャッシュクリア"):
                        st.session_state.pop('news_win_cache', None)
                        st.session_state.pop('news_stats', None)
                        st.info("news_win_cache をクリアしました。")
                if st.button("EVフル再計算 (キャッシュ無効化)"):
                    st.session_state.pop('ev_diff_cache', None)
                    st.success("ev_diff_cache を削除しました。次回は full rebuild になります。")
                if st.button("EV統計リセット"):
                    st.session_state.pop('ev_stats', None)
                    st.info("ev_stats をリセットしました。")
                st.session_state.setdefault('level_ttl_bars', 800)
                ttl_new = st.number_input("レベルTTL (bars)", 100, 5000, int(st.session_state['level_ttl_bars']), 50,
                                           help="最後にトレード発生した bar からこれを超えると集計から削除")
                if ttl_new != st.session_state['level_ttl_bars']:
                    st.session_state['level_ttl_bars'] = int(ttl_new)
                    st.info(f"TTL を {ttl_new} に更新。次の差分サイクルで適用。")
                if st.button("手動Prune実行"):
                    cache = st.session_state.get('ev_diff_cache')
                    if cache:
                        psig, li, bl, agg = cache
                        ttl = int(st.session_state['level_ttl_bars'])
                        to_del=[]
                        for k, vals in list(agg.items()):
                            if len(vals) < 3:
                                continue
                            _s,_c,lh = vals
                            if li - lh > ttl:
                                to_del.append(k)
                        for k in to_del:
                            del agg[k]
                        st.session_state['ev_diff_cache'] = (psig, li, bl, agg)
                        st.success(f"Prune: {len(to_del)} 件削除")

            def get_expected_for(level, direction):
                if ev_table is not None and not ev_table.empty:
                    sub = ev_table[(ev_table["level"]==float(level)) & (ev_table["dir"]==direction)]
                    if not sub.empty and int(sub["n"].iloc[0]) >= ev_level_min_samples:
                        return float(sub["avg"].iloc[0]), int(sub["n"].iloc[0])
                if ev_dir is not None and not ev_dir.empty:
                    subd = ev_dir[ev_dir["dir"]==direction]
                    if not subd.empty:
                        return float(subd["avg"].iloc[0]), int(subd["n"].iloc[0])
                return 0.0, 0


            # 期待値ランキングの計算部（該当箇所）
            exp_rows=[]
            for _, r in prob_df.iterrows():
                lv = float(r["level"])
                e_up, n_up = get_expected_for(lv, "long")
                e_dn, n_dn = get_expected_for(lv, "short")

                e_up_net = e_up - extra_cost_pips
                e_dn_net = e_dn - extra_cost_pips
                ev_up = r["P_up"] * e_up_net
                ev_dn = r["P_dn"] * e_dn_net

                BUY = "BUY"
                SELL = "SELL"
                # 最良の方向と期待値を決定
                best_dir = BUY if ev_up >= ev_dn else SELL
                best_ev  = ev_up if ev_up >= ev_dn else ev_dn
                # Risk guard 判定（レベル単位ではなくグローバル判定。必要なら方向別で2回呼び出し可）
                risk_allowed = True
                risk_reason = "ok"
                tg = st.session_state.get("trade_guard")
                if tg is not None:
                    # セッション名推定（既存 meta の theta_by_session 形式に倣う）
                    now_jst = pd.Timestamp.now(tz="Asia/Tokyo")
                    h = now_jst.hour
                    if 9 <= h < 15:
                        sess = "Tokyo"
                    elif 16 <= h < 24:
                        sess = "London"
                    elif h >= 22 or h < 5:
                        sess = "NY"
                    else:
                        sess = None
                    allow, msg = tg.allow_new_trade(ts=now_jst, session=sess)
                    risk_allowed = bool(allow)
                    risk_reason = msg
                reason_map = {
                    "ok": "許可",
                    "in_cooldown": "クールダウン中",
                    "daily_trade_limit": "日次上限到達",
                    "session_trade_limit": "セッション上限到達",
                    "atr_spike_halt": "ボラ急変停止",
                }
                risk_reason_jp = reason_map.get(risk_reason, risk_reason)
                exp_rows.append({
                    "level": lv,
                    "P_up": r.get("P_up", 0),
                    "P_dn": r.get("P_dn", 0),
                    "E_pips_up": e_up_net,  # コスト控除後の平均pips
                    "E_pips_dn": e_dn_net,
                    "EV_up": ev_up,
                    "EV_dn": ev_dn,
                    "best_action": best_dir,
                    "best_EV": best_ev,
                    "samples_up": n_up,
                    "samples_dn": n_dn,
                    "cost_pips": extra_cost_pips,  # 透明性のため列に残す
                    "risk_allowed": risk_allowed,
                    "risk_reason": risk_reason,
                    "risk_reason_jp": risk_reason_jp,
                })

            # 期待値ランキング表の表示
            ev_df = (pd.DataFrame(exp_rows)
                       .sort_values("best_EV", ascending=False)
                       .reset_index(drop=True))

            cols_show = ["level","P_up","P_dn","E_pips_up","E_pips_dn","EV_up","EV_dn","best_action","best_EV","samples_up","samples_dn","risk_allowed","risk_reason_jp"]
            st.dataframe(
                ev_df[cols_show]
                    .style.format({
                        "level":"{:.3f}",
                        "P_up":"{:.1%}","P_dn":"{:.1%}",
                        "E_pips_up":"{:.2f}","E_pips_dn":"{:.2f}",
                        "EV_up":"{:.2f}","EV_dn":"{:.2f}",
                        "best_EV":"{:.2f}"
                    })
            )
            if 'ev_meta' in locals():
                status = 'cache hit' if ev_meta.get('cache_hit') else 'recompute'
                if ev_meta.get('skipped'):
                    status += ' (skipped)'
                extra = ''
                if ev_meta.get('reason'):
                    extra = f" | {ev_meta['reason']}"
                st.caption(f"EV計算: {status} / {ev_meta.get('elapsed',0):.4f}s{extra}")

            # ---- Pending（リテスト待ち）管理・表示 ----
            st.session_state.setdefault('pending_trades', [])
            # 設定（軽量UIパラメタ）
            st.session_state.setdefault('pending_merge_eps_pips', 1.0)  # マージ閾値（pips）
            st.session_state.setdefault('pending_ttl_extra_bars', 0)    # 追加TTL（bars）
            st.session_state.setdefault('enable_sl_tp', False)
            st.session_state.setdefault('sl_pips', 10.0)
            st.session_state.setdefault('tp_pips', 20.0)
            st.session_state.setdefault('enable_trailing', False)
            st.session_state.setdefault('trail_pips', 15.0)
            st.session_state.setdefault('pending_strict_close', False)   # 二段条件: ヒゲ到達 AND 終値近接
            # ATR連動オプション
            st.session_state.setdefault('merge_eps_atr', False)
            st.session_state.setdefault('merge_eps_k', 0.0)
            st.session_state.setdefault('ttl_atr', False)
            st.session_state.setdefault('ttl_k', 0.0)
            st.session_state.setdefault('sl_tp_atr', False)
            st.session_state.setdefault('sl_k', 0.0)
            st.session_state.setdefault('tp_k', 0.0)
            st.session_state.setdefault('trail_atr', False)
            st.session_state.setdefault('trail_k', 0.0)

            def _ensure_pending(level: float, direction: str, open_i: int, expiry_i: int):
                # 既存の同一レベル・方向の pending/open は重複登録しない
                for p in st.session_state['pending_trades']:
                    if float(p.get('level')) == float(level) and p.get('dir') == direction and p.get('status') in ('pending','open'):
                        return
                st.session_state['pending_trades'].append({
                    'level': float(level), 'dir': direction, 'status': 'pending',
                    'open_i': int(open_i), 'expiry_i': int(expiry_i),
                    'created_i': int(open_i)
                })

            def _merge_pending(eps_pips: float):
                """同一方向・近接レベルの pending をマージ（古い方を優先）。open は対象外。"""
                lst = st.session_state['pending_trades']
                if not lst: return
                pv = pip_value("USDJPY")
                eps = float(eps_pips) * pv
                kept = []
                for p in sorted(lst, key=lambda x: (x.get('status')!='pending', x.get('created_i', 1<<30))):
                    if p.get('status') != 'pending':
                        kept.append(p); continue
                    merged = False
                    for q in kept:
                        if q.get('status')=='pending' and q.get('dir')==p.get('dir') and abs(float(q.get('level'))-float(p.get('level'))) <= eps:
                            # 古い q を残し、p を捨てる
                            merged = True
                            break
                    if not merged:
                        kept.append(p)
                st.session_state['pending_trades'] = kept

            # 現在バーでの更新処理
            i_cur = len(df) - 1
            if i_cur >= 1:
                t_now = df.index[-1]
                c_now = float(df['close'].iloc[-1])
                h_now = float(df['high'].iloc[-1]); l_now = float(df['low'].iloc[-1])
                l_prev = float(df['low'].iloc[-2]); h_prev = float(df['high'].iloc[-2])
                # ATR (pips)
                pv_local = pip_value("USDJPY")
                try:
                    atr_val = float(atr(df, 14).iloc[-1]) if len(df) >= 14 else float(df['high'].iloc[-1] - df['low'].iloc[-1])
                except Exception:
                    atr_val = float(df['high'].iloc[-1] - df['low'].iloc[-1])
                atr_pips = atr_val / pv_local if pv_local else 0.0
                # 実効パラメタ（ATR連動適用）
                eff_merge_eps_pips = float(st.session_state['pending_merge_eps_pips']) + (float(st.session_state.get('merge_eps_k',0.0))*atr_pips if st.session_state.get('merge_eps_atr') else 0.0)
                eff_ttl_extra = int(st.session_state['pending_ttl_extra_bars']) + (int(st.session_state.get('ttl_k',0.0) * atr_pips) if st.session_state.get('ttl_atr') else 0)
                eff_sl_pips = float(st.session_state['sl_pips']) + (float(st.session_state.get('sl_k',0.0))*atr_pips if st.session_state.get('sl_tp_atr') else 0.0)
                eff_tp_pips = float(st.session_state['tp_pips']) + (float(st.session_state.get('tp_k',0.0))*atr_pips if st.session_state.get('sl_tp_atr') else 0.0)
                eff_trail_pips = float(st.session_state['trail_pips']) + (float(st.session_state.get('trail_k',0.0))*atr_pips if st.session_state.get('trail_atr') else 0.0)

                # ニュース抑制（現在時点）
                if apply_news_filter:
                    _win_now = _get_news_windows_until_cached(t_now, news_imp_min)
                    suppressed_now = _is_suppressed_at(t_now, _win_now, news_win, news_imp_min, "水平線ブレイク(終値)")
                else:
                    suppressed_now = False

                # 期限切れ処理
                for p in st.session_state['pending_trades']:
                    if p.get('status') == 'pending' and i_cur > int(p.get('expiry_i', -1)):
                        p['status'] = 'expired'
                        p['closed_i'] = i_cur

                # 新規ブレイク → pending 追加（K>0 のとき）
                if not suppressed_now and int(retest_wait_k) > 0:
                    for lv in levels:
                        up_break = (c_now > lv + break_buffer) and (l_prev <= lv)
                        dn_break = (c_now < lv - break_buffer) and (h_prev >= lv)
                        if up_break:
                            extra = int(eff_ttl_extra)
                            _ensure_pending(lv, 'long', i_cur, i_cur + int(retest_wait_k) + extra)
                        if dn_break:
                            extra = int(eff_ttl_extra)
                            _ensure_pending(lv, 'short', i_cur, i_cur + int(retest_wait_k) + extra)

                # 近接 pending のマージ
                _merge_pending(eff_merge_eps_pips)

                # リテスト成立（現在バーで指値到達）→ open
                for p in st.session_state['pending_trades']:
                    if p.get('status') == 'pending' and i_cur >= int(p.get('open_i', 0)) and i_cur <= int(p.get('expiry_i', 0)):
                        lv = float(p['level'])
                        if not suppressed_now:
                            hit = False
                            if p.get('dir') == 'long':
                                hit = (l_now <= lv + float(touch_buffer))
                            else:
                                hit = (h_now >= lv - float(touch_buffer))
                            if st.session_state.get('pending_strict_close'):
                                hit = hit and (abs(c_now - lv) <= float(touch_buffer))
                            if hit:
                                p['status'] = 'open'
                                p['entry_i'] = i_cur
                                p['entry_price'] = c_now
                                p['exit_i'] = i_cur + int(fwd_n)

                # open → closed（決済）
                st.session_state.setdefault('pending_history', [])
                pv = pip_value("USDJPY")
                for p in st.session_state['pending_trades']:
                    if p.get('status') == 'open':
                        ex_i = int(p.get('exit_i', 1<<30))
                        direction = p.get('dir')
                        entry_price = float(p.get('entry_price', c_now))
                        # SL/TP/Trailing の早期クローズ判定
                        reason = None
                        if st.session_state.get('enable_sl_tp'):
                            sl_p = float(eff_sl_pips) * pv
                            tp_p = float(eff_tp_pips) * pv
                            if direction == 'long':
                                if l_now <= entry_price - sl_p:
                                    reason = 'sl'
                                elif h_now >= entry_price + tp_p:
                                    reason = 'tp'
                            else:
                                if h_now >= entry_price + sl_p:
                                    reason = 'sl'
                                elif l_now <= entry_price - tp_p:
                                    reason = 'tp'
                        if reason is None and st.session_state.get('enable_trailing'):
                            trail_p = float(eff_trail_pips) * pv
                            # 最高/最低有利値を更新
                            if direction == 'long':
                                best = float(p.get('best_high', entry_price))
                                best = max(best, h_now)
                                p['best_high'] = best
                                trail_stop = best - trail_p
                                p['trail_stop'] = trail_stop
                                if l_now <= trail_stop:
                                    reason = 'trail'
                            else:
                                best = float(p.get('best_low', entry_price))
                                best = min(best, l_now)
                                p['best_low'] = best
                                trail_stop = best + trail_p
                                p['trail_stop'] = trail_stop
                                if h_now >= trail_stop:
                                    reason = 'trail'
                        # 規定の exit_i 到達 or ルール成立でクローズ
                        do_close = False
                        exit_price = c_now
                        if reason is not None:
                            do_close = True
                        elif ex_i < len(df) and i_cur >= ex_i:
                            exit_price = float(df['close'].iloc[ex_i])
                            reason = 'time'
                            do_close = True
                        if do_close:
                            if direction == 'long':
                                realized = (exit_price - entry_price)/pv - float(spread_pips)
                            else:
                                realized = (entry_price - exit_price)/pv - float(spread_pips)
                            p['status'] = 'closed'
                            p['closed_i'] = i_cur if reason!='time' else ex_i
                            p['exit_price'] = exit_price
                            p['realized_pips'] = float(realized)
                            p['close_reason'] = reason
                            st.session_state['pending_history'].append({
                                'level': float(p['level']), 'dir': direction,
                                'entry_i': int(p.get('entry_i', i_cur)), 'exit_i': p.get('closed_i'),
                                'entry_price': entry_price, 'exit_price': exit_price,
                                'realized_pips': float(realized), 'reason': reason,
                            })

            # 表示用テーブル組立
            extra_cost = float(extra_cost_pips) if 'extra_cost_pips' in locals() else 0.0
            pend_rows = []
            def _get_prob_for(level: float, direction: str):
                if prob_df is None or prob_df.empty:
                    return float('nan')
                sub = prob_df[prob_df['level'] == float(level)]
                if sub.empty:
                    return float('nan')
                return float(sub['P_up' if direction=='long' else 'P_dn'].iloc[0])

            for p in st.session_state['pending_trades']:
                if p.get('status') in ('pending','open'):
                    lv = float(p['level']); direction = p['dir']
                    E, N = get_expected_for(lv, 'long' if direction=='long' else 'short')
                    E_net = (E - extra_cost) if E is not None else 0.0
                    P = _get_prob_for(lv, direction)
                    EV_ref = (P * E_net) if (not np.isnan(P)) else float('nan')
                    pend_rows.append({
                        'level': lv,
                        'dir': direction,
                        'status': p.get('status'),
                        'since_bars': max(0, i_cur - int(p.get('open_i', i_cur))) if i_cur is not None else None,
                        'bars_left': max(0, int(p.get('expiry_i', i_cur)) - i_cur) if i_cur is not None and p.get('status')=='pending' else None,
                        'entry_i': p.get('entry_i'),
                        'exit_i': p.get('exit_i'),
                        'E_pips_ref': E_net,
                        'P_dir_ref': P,
                        'EV_ref': EV_ref,
                        'samples': N,
                    })

            with st.expander("Pending（リテスト待ち）", expanded=False):
                if pend_rows:
                    _pdf = pd.DataFrame(pend_rows)
                    st.dataframe(
                        _pdf[["level","dir","status","since_bars","bars_left","E_pips_ref","P_dir_ref","EV_ref","samples"]]
                            .style.format({
                                "level": "{:.3f}", "E_pips_ref": "{:.2f}", "P_dir_ref": "{:.1%}", "EV_ref": "{:.2f}"
                            })
                    )
                else:
                    st.caption("現在、保有中のPendingはありません。")
                cols = st.columns(3)
                if cols[0].button("Pending全クリア"):
                    st.session_state['pending_trades'] = []
                    st.info("Pending をクリアしました。")
                # ポリシー調整
                st.write("— Pending管理ポリシー —")
                pm = cols[1].number_input("マージ閾値(pips)", 0.0, 10.0, float(st.session_state.get('pending_merge_eps_pips', 1.0)), 0.1)
                pe = cols[2].number_input("追加TTL(bars)", 0, 500, int(st.session_state.get('pending_ttl_extra_bars', 0)), 1)
                if pm != st.session_state['pending_merge_eps_pips']:
                    st.session_state['pending_merge_eps_pips'] = float(pm)
                    st.info("次バーからマージ閾値を適用します。")
                if pe != st.session_state['pending_ttl_extra_bars']:
                    st.session_state['pending_ttl_extra_bars'] = int(pe)
                    st.info("次のpending生成から追加TTLを適用します。")
                strict = st.checkbox("二段条件: ヒゲ到達 AND 終値近接", value=bool(st.session_state.get('pending_strict_close', False)))
                st.session_state['pending_strict_close'] = bool(strict)
                st.write("— ATR連動 —")
                cA, cB = st.columns(2)
                st.session_state['merge_eps_atr'] = bool(cA.checkbox("マージ閾値をATR連動", value=bool(st.session_state.get('merge_eps_atr', False))))
                st.session_state['merge_eps_k'] = float(cB.number_input("merge_eps_k", 0.0, 5.0, float(st.session_state.get('merge_eps_k', 0.0)), 0.1))
                cC, cD = st.columns(2)
                st.session_state['ttl_atr'] = bool(cC.checkbox("TTL追加をATR連動", value=bool(st.session_state.get('ttl_atr', False))))
                st.session_state['ttl_k'] = float(cD.number_input("ttl_k", 0.0, 50.0, float(st.session_state.get('ttl_k', 0.0)), 1.0))
                st.write("— 決済ルール —")
                en_st = st.checkbox("SL/TPを有効化", value=bool(st.session_state.get('enable_sl_tp', False)))
                st.session_state['enable_sl_tp'] = bool(en_st)
                c1, c2, c3 = st.columns(3)
                st.session_state['sl_pips'] = float(c1.number_input("SL(pips)", 0.0, 100.0, float(st.session_state.get('sl_pips', 10.0)), 0.5))
                st.session_state['tp_pips'] = float(c2.number_input("TP(pips)", 0.0, 200.0, float(st.session_state.get('tp_pips', 20.0)), 0.5))
                en_tr = c3.checkbox("トレーリング有効", value=bool(st.session_state.get('enable_trailing', False)))
                st.session_state['enable_trailing'] = bool(en_tr)
                st.session_state['trail_pips'] = float(st.number_input("トレーリング幅(pips)", 0.0, 200.0, float(st.session_state.get('trail_pips', 15.0)), 0.5))
                cE, cF = st.columns(2)
                st.session_state['sl_tp_atr'] = bool(cE.checkbox("SL/TPをATR連動", value=bool(st.session_state.get('sl_tp_atr', False))))
                st.session_state['sl_k'] = float(cF.number_input("sl_k", 0.0, 5.0, float(st.session_state.get('sl_k', 0.0)), 0.1))
                cG, cH = st.columns(2)
                st.session_state['tp_k'] = float(cG.number_input("tp_k", 0.0, 10.0, float(st.session_state.get('tp_k', 0.0)), 0.1))
                st.session_state['trail_atr'] = bool(cH.checkbox("トレールをATR連動", value=bool(st.session_state.get('trail_atr', False))))
                st.session_state['trail_k'] = float(st.number_input("trail_k", 0.0, 10.0, float(st.session_state.get('trail_k', 0.0)), 0.1))
                # 直近のクローズド履歴
                hist = st.session_state.get('pending_history', [])
                if hist:
                    hdf = pd.DataFrame(hist)
                    st.write("最近のクローズド（直近50件）")
                    st.dataframe(
                        hdf.tail(50)[["level","dir","entry_i","exit_i","entry_price","exit_price","realized_pips"]]
                           .style.format({
                               "level": "{:.3f}", "entry_price": "{:.3f}", "exit_price": "{:.3f}", "realized_pips": "{:.2f}"
                           })
                    )
                    # サマリ
                    rp = hdf["realized_pips"].astype(float)
                    avgp = float(rp.mean()) if not rp.empty else 0.0
                    winr = float((rp > 0).mean()) if not rp.empty else 0.0
                    st.caption(f"Closed合計: {len(hdf)} | 平均pips: {avgp:.2f} | 勝率: {winr:.1%}")
                    if cols[1].button("履歴クリア"):
                        st.session_state['pending_history'] = []
                        st.info("履歴をクリアしました。")

# === 予測ブロック直後（P_up/P_dn を集計する箇所の後）に確率バッファ・PSI・θ超過率集計 ===
def init_session_state_buffer():
    if "prob_buffer" not in st.session_state:
        st.session_state.prob_buffer = []

# ...existing code...

    # === 予測ブロック直後（P_up/P_dn を集計する箇所の後）に確率バッファ・PSI・θ超過率集計 ===
    init_session_state_buffer()

    curr_probs = update_prob_buffer(prob_df)
    psi_val, sev, ex_rate, kl, js, hell = calc_psi_and_exrate(curr_probs, baseline_probs, theta_up, theta_dn)
    st.session_state['drift_meta'] = {
        'psi': psi_val, 'severity': sev, 'ex_rate': ex_rate,
        'kl': kl, 'js': js, 'hellinger': hell,
    }

    # 価格の最終時刻（JST想定のindex）を取得
    try:
        last_ts = df.index[-1].tz_convert("UTC")
    except Exception:
        last_ts = None

    hr = healthcheck(
        model_path="models/break_model.joblib",
        meta_path="models/break_meta.json",
        windows_df=windows_df,
        last_price_ts=last_ts,  # None可
        max_age_min=5
    )
    st.write(f"**overall**: {'✅OK' if hr.ok else '⚠️CHECK'}  |  price_age: {hr.details.get('price_feed','?')}")
    st.write(f"- model: {hr.details.get('model')}")
    st.write(f"- meta : {hr.details.get('meta')}")
    st.write(f"- events: {hr.details.get('event_windows')}")

# === 予測ブロック直後（P_up/P_dn を集計する箇所の後）に確率バッファ・PSI・θ超過率集計 ===

# ...existing code...

    with st.sidebar.expander("🩺 Health & Drift", expanded=True):
        st.write(f"**overall**: {'✅OK' if hr.ok else '⚠️CHECK'}  |  price_age: {hr.details.get('price_feed','?')}")
        st.write(f"- model: {hr.details.get('model')}")
        st.write(f"- meta : {hr.details.get('meta')}")
        st.write(f"- events: {hr.details.get('event_windows')}")
        st.write("---")
        st.write(f"**PSI**: {psi_val:.3f}  ({sev})")
        st.progress(min(max((0.3 - float(psi_val if not math.isnan(psi_val) else 0))/0.3, 0.0), 1.0))
        st.caption("PSI<0.10:安定 / <0.25:注意 / ≥0.25:ドリフト")
        st.write(f"**θ exceed rate (now)**: {ex_rate:.2%}" if not np.isnan(ex_rate) else "θ exceed rate: n/a")

    ev_df = (pd.DataFrame(exp_rows)
               .sort_values("best_EV", ascending=False)
               .reset_index(drop=True))

    st.dataframe(
        ev_df[["level","P_up","P_dn","E_pips_up","E_pips_dn","EV_up","EV_dn","best_action","best_EV","samples_up","samples_dn"]]
            .style.format({
                "level":"{:.3f}",
                "P_up":"{:.1%}","P_dn":"{:.1%}",
                "E_pips_up":"{:.2f}","E_pips_dn":"{:.2f}",
                "EV_up":"{:.2f}","EV_dn":"{:.2f}",
                "best_EV":"{:.2f}"
            })
    )
    st.caption(
        f"H（学習）: {meta.get('H','?')} / H（表示）: {break_prob_h} / "
        f"break_buffer(学習): {meta.get('break_buffer','?')} / touch_buffer(学習): {meta.get('touch_buffer','?')}  |  "
        f"Calibration: { 'ON' if meta.get('calibrated') else 'OFF' }"
    )
    st.caption(
        f"※ 期待pipsは『水平線ブレイク(終値)』のバックテスト平均から推定"
    )

if show_ev_rank and not show_break_prob:
    st.info("期待値ランキングは「今からの水平線ブレイク確率」をONにすると表示されます。")

# ---------------- メタ統合サマリ（EV/θ/ニュース/ペンディング） ----------------
with st.expander("メタ統合サマリ", expanded=False):
    # EVメタ
    evs = st.session_state.get('ev_stats', {})
    ev_mode = evs.get('last_mode')
    ev_ms = evs.get('last_elapsed_ms')
    ev_added = evs.get('last_added_trades')
    ev_agg = evs.get('agg_size')
    ev_pruned = evs.get('pruned_last')
    # θメタ
    theta_cur = st.session_state.get('theta_current')
    use_adapt = st.session_state.get('use_adaptive_theta')
    # ニュースキャッシュ
    nstats = st.session_state.get('news_stats', {})
    nh, nm, ns = nstats.get('hits'), nstats.get('miss'), nstats.get('size')
    # Pending
    pend = st.session_state.get('pending_trades', [])
    pend_open = sum(1 for p in pend if p.get('status') == 'open')
    pend_wait = sum(1 for p in pend if p.get('status') == 'pending')
    phist = st.session_state.get('pending_history', [])
    phn = len(phist)
    avgp = None
    try:
        import pandas as _pd
        if phist:
            rp = _pd.Series([float(x.get('realized_pips', float('nan'))) for x in phist])
            avgp = float(rp.mean()) if not rp.empty else None
    except Exception:
        avgp = None
    # ドリフト（PSI/KL/JS/Hellinger）とガード状態
    drift = st.session_state.get('drift_meta', {})
    guard_state = {}
    try:
        tg = st.session_state.get('trade_guard')
        if tg is not None:
            guard_state = tg.snapshot() if hasattr(tg, 'snapshot') else {}
    except Exception:
        guard_state = {}
    st.write({
        'EV': {'mode': ev_mode, 'elapsed_ms': ev_ms, 'added': ev_added, 'agg_size': ev_agg, 'pruned_last': ev_pruned},
        'Theta': {'adaptive_on': use_adapt, 'theta_current': theta_cur},
        'NewsCache': {'hits': nh, 'miss': nm, 'size': ns},
        'Pending': {'open': pend_open, 'waiting': pend_wait, 'closed_total': phn, 'closed_avg_pips': avgp},
        'Drift': drift,
        'Guard': guard_state
    })

# ---------------- パターンの「次の動きの予測」一覧（測定値ターゲット & EV） ----------------
st.subheader("🧭 パターン検出：次の動きの予測（測定値ターゲット & EV）")
if not patterns:
    st.info("現在、検出されたパターンはありません。lookback/感度を調整してください。")
else:
    ev_table, ev_dir = compute_expected_pips_table_for_levels(
        df, levels, fwd_n, break_buffer, spread_pips,
        news_df, news_win, news_imp_min, apply_news_filter,
        touch_buffer, retest_wait_k
    )
    model, Xcols, meta = load_break_model(prob_model_path)

    rows=[]
    ts_now = df.index[-1]
    for p in patterns:
        meas = measured_targets(p)
        tgt_up_price = df["close"].iloc[-1] + meas["up"]
        tgt_dn_price = df["close"].iloc[-1] + meas["down"]

        upper_level = None; lower_level = None
        kind = p.get('kind') if isinstance(p, dict) else getattr(p, 'kind', None)
        params = p['params'] if isinstance(p, dict) else getattr(p, 'params', {})
        if kind and kind.startswith("triangle"):
            subp = df.loc[params["sub_start"]:params["sub_end"]]
            m1,b1 = params["upper"]; m2,b2 = params["lower"]
            x_now  = len(subp)-1
            y_u = float(m1*x_now + b1); y_l = float(m2*x_now + b2)
            upper_level, lower_level = y_u, y_l
        elif kind=="rectangle":
            upper_level, lower_level = float(params["upper"]), float(params["lower"])
        elif kind=="double_top":
            upper_level, lower_level = None, float(params["neck"])
        elif kind=="double_bottom":
            upper_level, lower_level = float(params["neck"]), None
        elif kind in ("flag_up","flag_dn","pennant"):
            upper_level, lower_level = float(params["upper_now"]), float(params["lower_now"])
        elif kind=="head_shoulders":
            upper_level, lower_level = None, float(params["neck"])
        elif kind=="inverse_head_shoulders":
            upper_level, lower_level = float(params["neck"]), None

        P_up = P_dn = np.nan
        if model is not None:
            if upper_level is not None:
                f_up = make_features_for_level(df, ts_now, upper_level, +1, touch_buffer)
                if isinstance(f_up, dict):
                    row_up = {k: f_up.get(k, 0.0) for k in use_cols}
                elif isinstance(f_up, list):
                    row_up = dict(zip(use_cols, f_up))
                else:
                    raise TypeError("f_up must be dict or list")
                row_up["timestamp"] = ts_now
                df_up = pd.DataFrame([row_up])
                for col in use_cols:
                    if col not in df_up.columns:
                        df_up[col] = 0.0
                df_up = df_up[use_cols]
                df_up["timestamp"] = ts_now
                pred_up = predict_with_session_theta(df_up, model, use_cols, meta)
                P_up = float(pred_up["proba"].iloc[0])
            if lower_level is not None:
                f_dn = make_features_for_level(df, ts_now, lower_level, -1, touch_buffer)
                if isinstance(f_dn, dict):
                    row_dn = {k: f_dn.get(k, 0.0) for k in use_cols}
                elif isinstance(f_dn, list):
                    row_dn = dict(zip(use_cols, f_dn))
                else:
                    raise TypeError("f_dn must be dict or list")
                row_dn["timestamp"] = ts_now
                df_dn = pd.DataFrame([row_dn])
                for col in use_cols:
                    if col not in df_dn.columns:
                        df_dn[col] = 0.0
                df_dn = df_dn[use_cols]
                df_dn["timestamp"] = ts_now
                pred_dn = predict_with_session_theta(df_dn, model, use_cols, meta)
                P_dn = float(pred_dn["proba"].iloc[0])

        def _ev_for(level, direction):
            if level is None: return (0.0, 0, np.nan)
            if ev_table is not None and not ev_table.empty:
                sublv = ev_table[(ev_table["dir"]==direction)]
                if not sublv.empty:
                    idx_min = (sublv["level"]-float(level)).abs().idxmin()
                    avg, n = float(sublv.loc[idx_min,"avg"]), int(sublv.loc[idx_min,"n"])
                    if n >= ev_level_min_samples:
                        p = P_up if direction=="long" else P_dn
                        avg_net = avg - extra_cost_pips
                        return (avg_net, n, (p*avg_net) if not np.isnan(p) else np.nan)
            if ev_dir is not None and not ev_dir.empty:
                dsub = ev_dir[ev_dir["dir"]==direction]
                if not dsub.empty:
                    avg, n = float(dsub["avg"].iloc[0]), int(dsub["n"].iloc[0])
                    p = P_up if direction=="long" else P_dn
                    avg_net = avg - extra_cost_pips
                    return (avg_net, n, (p*avg_net) if not np.isnan(p) else np.nan)
            return (0.0, 0, np.nan)

        E_up, N_up, EV_up = _ev_for(upper_level, "long")
        E_dn, N_dn, EV_dn = _ev_for(lower_level, "short")

        # 予測値のデバッグ出力
        print(f"[DEBUG] upper_level={upper_level}, lower_level={lower_level}")
        print(f"[DEBUG] df_up: {df_up if 'df_up' in locals() else None}")
        print(f"[DEBUG] df_dn: {df_dn if 'df_dn' in locals() else None}")
        print(f"[DEBUG] P_up={P_up}, P_dn={P_dn}")
        kind = p.get('kind') if isinstance(p, dict) else getattr(p, 'kind', None)
        quality = p.get('quality') if isinstance(p, dict) else getattr(p, 'quality', None)
        rows.append(dict(
            pattern=kind, quality=round(quality,1) if quality is not None else None,
            upper_level=upper_level, lower_level=lower_level,
            P_up=P_up, P_dn=P_dn,
            E_pips_up=E_up, E_pips_dn=E_dn,
            EV_up=EV_up, EV_dn=EV_dn,
            target_up=tgt_up_price, target_dn=tgt_dn_price
        ))

    out = pd.DataFrame(rows)
    if not out.empty:
        out["best_action"] = out.apply(lambda r: "BUY" if (pd.notna(r["EV_up"]) and (r["EV_up"]>= (r["EV_dn"] if pd.notna(r["EV_dn"]) else -1e9))) else "SELL", axis=1)
        out["best_EV"] = out.apply(lambda r: r["EV_up"] if r["best_action"]=="BUY" else r["EV_dn"], axis=1)
        # 数値カラムのみ型変換してフォーマット、str型には適用しない
        num_cols = ["upper_level","lower_level","P_up","P_dn","E_pips_up","E_pips_dn","EV_up","EV_dn","best_EV","target_up","target_dn"]
        for col in num_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        st.dataframe(
            out[["pattern","quality","upper_level","lower_level","P_up","P_dn",
                 "E_pips_up","E_pips_dn","EV_up","EV_dn","best_action","best_EV",
                 "target_up","target_dn"]]
            .style.format({
                "upper_level":"{:.3f}", "lower_level":"{:.3f}",
                "P_up":"{:.1%}", "P_dn":"{:.1%}",
                "E_pips_up":"{:.2f}", "E_pips_dn":"{:.2f}",
                "EV_up":"{:.2f}", "EV_dn":"{:.2f}", "best_EV":"{:.2f}",
                "target_up":"{:.3f}", "target_dn":"{:.3f}",
            })
        )
        st.caption("※ P_up/P_dn は学習モデル（水平ブレイク）を、パターンの境界（上辺/下辺/ネック/チャネル）に当てはめて推定。ターゲットは測定値（旗竿・厚み・ヘッド高さ 等）。")
    else:
        st.info("パターンは検出されましたが、テーブル化できる十分な指標がありません。")

st.markdown("---")
st.markdown("""
## 免責事項 / Disclaimer

本アプリは、過去の価格データ等をもとに作成された分析ツール・教育用コンテンツであり、
金融商品取引法に基づく投資助言・代理業務を行うものではありません。

本アプリが提供する情報・シグナル・予測結果は、将来の成果や利益を保証するものではなく、
投資判断はすべて利用者ご自身の責任において行ってください。

当方は、本アプリの利用により生じたいかなる損失・損害についても、一切の責任を負いません。

また、本アプリが使用するデータは Yahoo Finance 等の外部サービスを通じて取得しています。
データの正確性・完全性は保証されませんのでご了承ください。

本アプリの利用により生じるあらゆるリスクは、利用者ご自身の自己責任においてご対応ください。
""")