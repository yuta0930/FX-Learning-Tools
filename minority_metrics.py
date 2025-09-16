

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from purged_cv import make_time_groups, PurgedGroupTimeSeriesSplit
from sweep_labels import export_label_audit_samples
from sklearn.metrics import (
    matthews_corrcoef,
    balanced_accuracy_score,
    f1_score,
    brier_score_loss,
    precision_recall_curve,
    auc,
)

def summarize_minor_metrics(y_true, y_pred_proba, threshold=0.5):
    # 2値分類前提
    y_pred = (y_pred_proba >= threshold).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    ba = balanced_accuracy_score(y_true, y_pred)
    f1_neg = f1_score(y_true, y_pred, pos_label=0)
    f1_pos = f1_score(y_true, y_pred, pos_label=1)
    brier = brier_score_loss(y_true, y_pred_proba)
    # 少数派クラス特定
    n0, n1 = np.sum(y_true==0), np.sum(y_true==1)
    minority_class = 0 if n0 < n1 else 1 if n1 < n0 else 1
    # PR曲線AUC
    pr_auc_pos = auc(*precision_recall_curve(y_true, y_pred_proba, pos_label=1)[1::-1])
    pr_auc_neg = auc(*precision_recall_curve(1-y_true, 1-y_pred_proba, pos_label=1)[1::-1])
    return {
        "MCC": mcc,
        "BA": ba,
        "F1_neg": f1_neg,
        "F1_pos": f1_pos,
        "Brier": brier,
        "minority_class": minority_class,
        "PR_AUC_pos": pr_auc_pos,
        "PR_AUC_neg": pr_auc_neg,
    }

if __name__ == "__main__":
    """少数派指標のクロスバリデーションレポートとラベル監査CSVを生成する簡易スクリプト。

    以前は下部にほぼ同一内容のコードブロックが重複して存在していたため削除しました。
    主ブロックでは 'timestamp' 列を時間カラムとして使用します。必要に応じて変更してください。
    """
    try:
        print("[START] Purged+Embargo CV・ラベル監査CSV・少数派KPI主指標化 実行開始")
        df = pd.read_csv("data/USDJPY_15m.csv")
        print("dfのカラム:", df.columns)
        # y が単一定数なら暫定で二分
        if df["y"].nunique() == 1:
            print("[INFO] 'y' が単一定数 -> 擬似的に 0/1 を半分ずつ割当")
            n = len(df)
            df.loc[: n // 2, "y"] = 0
            df.loc[n // 2 :, "y"] = 1
            df.to_csv("data/USDJPY_15m.csv", index=False)
        # ラベル監査サンプル抽出
        pos_csv, neg_csv = export_label_audit_samples(
            df, label_col="y", proba_col=None, out_prefix="audit_break"
        )
        print(f"[label audit] exported: {pos_csv}, {neg_csv}")
        tcol = "timestamp"
        assert tcol in df.columns, f"dfに '{tcol}' 列がありません"
        assert "y" in df.columns, "dfに 'y' 列がありません"
        feature_cols = [c for c in df.columns if c not in [tcol, "y"]]
        assert feature_cols, "特徴量となるカラムがありません"
        X = df[feature_cols]
        y = df["y"]
        groups = make_time_groups(df[tcol], freq="D")
        cv = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=1, embargo_groups=1)
        cv_rows = []
        for fold, (tr_idx, te_idx) in enumerate(cv.split(df, groups=groups), 1):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx].values
            X_te, y_te = X.iloc[te_idx], y.iloc[te_idx].values
            model = LogisticRegression(max_iter=200)
            model.fit(X_tr, y_tr)
            p_te = model.predict_proba(X_te)[:, 1]
            row = summarize_minor_metrics(y_te, p_te)
            row["fold"] = fold
            row["pos_ratio_fold"] = float(np.mean(y_te == 1))
            cv_rows.append(row)
            print(
                f"[CV purged] fold={fold} MCC={row['MCC']:.3f} BA={row['BA']:.3f} "
                f"F1_neg={row['F1_neg']:.3f} PR_minor="
                f"{(row['PR_AUC_pos'] if row['minority_class']==1 else row['PR_AUC_neg']):.3f} "
                f"Brier={row['Brier']:.4f} pos_ratio={row['pos_ratio_fold']:.3f}"
            )

        cv_df = pd.DataFrame(cv_rows)
        cv_df.to_csv("cv_minor_summary.csv", index=False)
        print("[CV SUMMARY] saved -> cv_minor_summary.csv")

        # EV の簡易計算例（プレースホルダ）
        p_ratio = float(np.mean(y == 1))
        reward_win = 1.0
        reward_lose = 1.0
        cost = 0.02
        ev_tr = p_ratio * reward_win - (1 - p_ratio) * reward_lose - cost
        print(
            f"[EV] net(costed)={ev_tr:.3f} | gross(no cost)={p_ratio*reward_win - (1-p_ratio)*reward_lose:.3f} | cost={cost:.3f}"
        )
        print("[END] Purged+Embargo CV・ラベル監査CSV・少数派KPI主指標化 実行完了")
    except FileNotFoundError:
        print("[ERROR] data/USDJPY_15m.csv が見つかりません。配置してください。")
    except AssertionError as e:
        print("[ERROR] カラム・データエラー:", e)
        if "y" in str(e):
            print("[INFO] 'y' 列を追加し 0 で初期化します。再実行してください。")
            try:
                df = pd.read_csv("data/USDJPY_15m.csv")
                df["y"] = 0
                df.to_csv("data/USDJPY_15m.csv", index=False)
            except Exception as e2:
                print("[ERROR] 'y' 列追加時に失敗:", e2)
    except Exception as e:
        import traceback
        print("[ERROR] その他のエラー:", e)
        traceback.print_exc()
        
    # NOTE: 下部に存在した重複コードブロック（ほぼ同一処理・'time' カラム使用版）は整理のため削除しました。
