import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Any, Optional

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

            # 健全性チェック：確率が常時同値になっていないこと
            if hard_assert and np.isclose(p_up, p_dn, atol=1e-12):
                raise RuntimeError(
                    f"[BUG] P_up == P_dn を検出。level={lv} ts={ts_i} p_up={p_up:.6f} p_dn={p_dn:.6f}"
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
