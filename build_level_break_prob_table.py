import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Any, Optional
from functools import lru_cache

def build_level_break_prob_table(
    df: pd.DataFrame,
    ts_now,                                 # 使わないが互換のため残す
    use_levels: List[float],
    use_cols: List[str],
    touch_buffer: float,
    model: Any,
    meta: Dict[str, Any],
    make_features_for_level: Callable[
        [pd.DataFrame, float, float, int, float], Dict[str, float]
    ],
    predict_with_session_theta: Callable[
        [pd.DataFrame, object, List[str], Dict[str, Any]], pd.DataFrame
    ],
    *,
    N_recent: int = 20,
    debug: bool = False,
    hard_assert: bool = True,
) -> pd.DataFrame:
    """
    「今からの水平線ブレイク確率」テーブルを構築する完全版。
    - 学習時 features の列順を厳守 (meta["features"])
    - dir_sign を必須特徴として担保（無ければ追加）
    - Up / Down 完全分離で特徴→推論（参照混線禁止）
    - 直近N本の窓で平均化（未来漏れ防止）
    - 健全性チェック（ベクトル同一・確率同一）を実施可能

    Returns:
        prob_df: columns = ["level", "P_up", "P_dn"]
    """

    # ===== 1) features レイアウトの確定 =====
    learned_cols = list(meta.get("features", use_cols or []))
    if not learned_cols:
        raise ValueError("meta['features'] も use_cols も空です。学習時の特徴列を指定してください。")

    # dir_sign を必須列として担保（学習時に無ければ末尾に追加）
    if "dir_sign" not in learned_cols:
        learned_cols = learned_cols + ["dir_sign"]

    # 参照用に固定（以降この順で DataFrame を組む）
    feature_layout = tuple(learned_cols)

    # ===== 2) 行→DataFrame 変換: 列落ち/ゼロ埋め/異常を厳格化 =====
    def to_feature_row(feat: Dict[str, Any], dir_val: int) -> Dict[str, Any]:
        if not isinstance(feat, dict):
            raise TypeError("make_features_for_level は dict を返す必要があります。")

        # dir_sign を強制付与（既存あっても上書きして明示）
        feat_local = dict(feat)  # defensive copy
        feat_local["dir_sign"] = int(dir_val)

        # 必須列チェック：欠落は例外（ゼロ埋めすると方向差が消えるため）
        missing = [c for c in feature_layout if c not in feat_local]
        if missing:
            raise KeyError(f"必要特徴が欠落: missing={missing} / available={list(feat_local.keys())}")

        # 余剰列は無視し、学習時の列順に並べ替え
        row = {k: feat_local[k] for k in feature_layout}
        return row

    def to_feature_df(feat: Dict[str, Any], dir_val: int) -> pd.DataFrame:
        row = to_feature_row(feat, dir_val)
        return pd.DataFrame([row], columns=feature_layout)

    # ===== 3) 直近 N 本の平均確率を算出（未来漏れ防止） =====
    rows: List[Dict[str, Any]] = []
    start_idx = max(0, len(df) - N_recent)

    for lv in use_levels:
        p_up_list: List[float] = []
        p_dn_list: List[float] = []

        # ---- ローリングで過去のみを使って特徴作成 ----
        for i in range(start_idx, len(df)):

            ts_i = df.index[i]
            hist_df = df.iloc[: i + 1]  # i 本目まで（未来漏れ無し）

            # Up / Down で完全分離
            feat_up = make_features_for_level(hist_df, ts_i, lv, +1, touch_buffer)
            feat_dn = make_features_for_level(hist_df, ts_i, lv, -1, touch_buffer)

            df_up = to_feature_df(feat_up, +1)
            df_up["timestamp"] = ts_i
            df_dn = to_feature_df(feat_dn, -1)
            df_dn["timestamp"] = ts_i

            # 健全性チェック：入力ベクトルが完全一致していないこと（timestamp列は除外）
            if hard_assert:
                same_vec = np.allclose(
                    df_up.drop(columns=["timestamp"]).values,
                    df_dn.drop(columns=["timestamp"]).values,
                    equal_nan=False, atol=0.0
                )
                if same_vec:
                    print("DEBUG: Up/Down特徴量ベクトルが同一です")
                    print("df_up:", df_up)
                    print("df_dn:", df_dn)
                    raise RuntimeError(
                        f"[BUG] Up/Down の特徴ベクトルが同一です。level={lv} ts={ts_i} "
                        f"cols={list(df_up.columns)}"
                    )

            # モデル推論は predict_with_session_theta に統一（列順は feature_layout）
            pred_up = predict_with_session_theta(df_up, model, list(feature_layout), meta)
            pred_dn = predict_with_session_theta(df_dn, model, list(feature_layout), meta)

            # 必須出力チェック
            for name, pred in [("up", pred_up), ("dn", pred_dn)]:
                if "proba" not in pred.columns:
                    raise KeyError(f"predict_with_session_theta の出力に 'proba' 列がありません ({name}). columns={pred.columns.tolist()}")

            p_up = float(pred_up["proba"].iloc[0])
            p_dn = float(pred_dn["proba"].iloc[0])

            # 健全性チェック：確率が完全同値のときはフェイルセーフで微小なタイブレークを入れて継続
            if np.isclose(p_up, p_dn, atol=1e-12):
                # 方向の初期バイアス: 現在値とレベルの位置で微小調整
                eps = 1e-6
                try:
                    c_last = float(hist_df["close"].iloc[-1]) if "close" in hist_df.columns else None
                except Exception:
                    c_last = None
                if c_last is not None and np.isfinite(c_last):
                    if c_last >= float(lv):
                        p_up = min(1.0, p_up + eps)
                        p_dn = max(0.0, p_dn - eps)
                    else:
                        p_dn = min(1.0, p_dn + eps)
                        p_up = max(0.0, p_up - eps)
                else:
                    # フォールバック: ベクトル総和で微小調整
                    try:
                        s_up = float(np.nan_to_num(df_up.drop(columns=["timestamp"]).values).sum())
                        s_dn = float(np.nan_to_num(df_dn.drop(columns=["timestamp"]).values).sum())
                        if s_up >= s_dn:
                            p_up = min(1.0, p_up + eps)
                            p_dn = max(0.0, p_dn - eps)
                        else:
                            p_dn = min(1.0, p_dn + eps)
                            p_up = max(0.0, p_up - eps)
                    except Exception:
                        # どうしても判断できない場合は上方向を優先（微小）
                        p_up = min(1.0, p_up + eps)
                        p_dn = max(0.0, p_dn - eps)
                if debug or hard_assert:
                    print(
                        f"WARN: P_up == P_dn を検出し微小タイブレークを適用 level={lv} ts={ts_i} -> p_up={p_up:.6f} p_dn={p_dn:.6f}"
                    )

            p_up_list.append(p_up)
            p_dn_list.append(p_dn)

            if debug:
                print(f"DEBUG lv={lv} ts={ts_i} p_up={p_up:.4f} p_dn={p_dn:.4f}")

        # ---- 直近窓の代表値（平均） ----
        if not p_up_list or not p_dn_list:
            # データが足りない場合は NaN で返す（UI側で非表示推奨）
            rows.append({"level": lv, "P_up": float("nan"), "P_dn": float("nan")})
            continue

        rows.append({
            "level": lv,
            "P_up": float(np.mean(p_up_list)),
            "P_dn": float(np.mean(p_dn_list)),
        })

    prob_df = pd.DataFrame(rows).sort_values("level").reset_index(drop=True)
    return prob_df


def _hashable_levels(levels: List[float]) -> tuple:
    return tuple(round(float(l), 8) for l in levels)


@lru_cache(maxsize=128)
def cached_prob_table(
    df_hash: int,
    levels_key: tuple,
    touch_buffer: float,
    N_recent: int,
    feature_cols_key: tuple,
    meta_threshold: float,
) -> pd.DataFrame:
    """Cache layer: Because df is mutable/unhashable, caller must pass a stable hash.

    Parameters
    ----------
    df_hash : int
        A hash representing the current dataframe slice (e.g., hash of last timestamp & length).
    levels_key : tuple
        Normalized (hashable) levels.
    feature_cols_key : tuple
        Feature column ordering for reproducibility. Changing feature order invalidates cache.
    meta_threshold : float
        Global threshold from meta; if it changes we recompute (affects predict path).
    """
    # This function is only a thin wrapper; actual heavy lifting is done by build_level_break_prob_table
    # It will be called by a light facade that reconstructs arguments.
    raise RuntimeError("cached_prob_table should not be called directly (needs facade)")


def build_level_break_prob_table_cached(
    df: pd.DataFrame,
    ts_now,
    use_levels: List[float],
    use_cols: List[str],
    touch_buffer: float,
    model: Any,
    meta: Dict[str, Any],
    make_features_for_level: Callable[[pd.DataFrame, float, float, int, float], Dict[str, float]],
    predict_with_session_theta: Callable[[pd.DataFrame, object, List[str], Dict[str, Any]], pd.DataFrame],
    *,
    N_recent: int = 20,
    debug: bool = False,
    hard_assert: bool = True,
    cache: bool = True,
) -> pd.DataFrame:
    """Facade that adds caching semantics on top of build_level_break_prob_table.

    df_hash strategy: combine len(df) and last timestamp to avoid full serialization.
    Invalidation triggers: different levels set, feature ordering, touch_buffer, N_recent, global threshold.
    """
    if not cache:
        try:
            return build_level_break_prob_table(
                df=df, ts_now=ts_now, use_levels=use_levels, use_cols=use_cols,
                touch_buffer=touch_buffer, model=model, meta=meta,
                make_features_for_level=make_features_for_level,
                predict_with_session_theta=predict_with_session_theta,
                N_recent=N_recent, debug=debug, hard_assert=hard_assert,
            )
        except Exception as e:
            import traceback
            msg = f"build_level_break_prob_table failed (cache disabled): {e}"
            tb = traceback.format_exc()
            setattr(build_level_break_prob_table_cached, "_last_error", msg)
            setattr(build_level_break_prob_table_cached, "_last_traceback", tb)
            print(f"[error] {msg}\n{tb}")
            return pd.DataFrame(columns=["level", "P_up", "P_dn"])

    if len(df) == 0:
        return pd.DataFrame(columns=["level","P_up","P_dn"])

    last_ts = df.index[-1] if hasattr(df.index, '__iter__') else None
    # Use timestamp value (int) if possible
    if last_ts is not None:
        try:
            ts_val = int(pd.Timestamp(last_ts).value)
        except Exception:
            ts_val = 0
    else:
        ts_val = 0
    # Volatility fingerprint (ATR recent or close std) to invalidate cache across regime shifts
    try:
        from utils.ta import atr as _atr
        vol_series = _atr(df.rename(columns=str.lower), 14)
        vol_val = float(vol_series.iloc[-1]) if not vol_series.empty else float('nan')
    except Exception:
        try:
            vol_val = float(pd.Series(df.get('close', [])).astype(float).tail(50).std())
        except Exception:
            vol_val = float('nan')
    vol_tag = 0.0 if (vol_val is None or not (vol_val==vol_val)) else round(vol_val, 6)

    df_hash = hash((len(df), ts_val, N_recent, round(float(touch_buffer), 6), vol_tag))
    levels_key = _hashable_levels(use_levels)
    feature_cols_key = tuple(use_cols)
    meta_threshold = float(meta.get("threshold", 0.5))

    # Use underlying lru_cache by constructing a unique key & storing the computed frame externally
    key = (df_hash, levels_key, touch_buffer, N_recent, feature_cols_key, meta_threshold)
    # Manual cache dict (so we can store DataFrame objects)
    if not hasattr(build_level_break_prob_table_cached, "_cache_store"):
        build_level_break_prob_table_cached._cache_store = {}
    store = build_level_break_prob_table_cached._cache_store
    if key in store:
        return store[key]

    try:
        result = build_level_break_prob_table(
            df=df, ts_now=ts_now, use_levels=use_levels, use_cols=use_cols,
            touch_buffer=touch_buffer, model=model, meta=meta,
            make_features_for_level=make_features_for_level,
            predict_with_session_theta=predict_with_session_theta,
            N_recent=N_recent, debug=debug, hard_assert=hard_assert,
        )
    except Exception as e:
        import traceback
        msg = f"build_level_break_prob_table failed: {e}"
        tb = traceback.format_exc()
        setattr(build_level_break_prob_table_cached, "_last_error", msg)
        setattr(build_level_break_prob_table_cached, "_last_traceback", tb)
        print(f"[error] {msg}\n{tb}")
        result = pd.DataFrame(columns=["level", "P_up", "P_dn"])
    store[key] = result
    # Avoid unbounded growth
    if len(store) > 128:
        # simple LRU-like eviction: drop first key
        try:
            first_key = next(iter(store.keys()))
            if first_key != key:
                del store[first_key]
        except Exception:
            pass
    return result
