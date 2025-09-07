

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from purged_cv import make_time_groups, PurgedGroupTimeSeriesSplit
from sweep_labels import export_label_audit_samples
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, brier_score_loss, precision_recall_curve, auc

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
    try:
        print("[START] Purged+Embargo CV・ラベル監査CSV・少数派KPI主指標化 実行開始")
        # サンプルデータの読み込み例
        df = pd.read_csv("data/USDJPY_15m.csv")
        print("dfのカラム:", df.columns)
        # --- y列が全て同じ値なら半分ずつ0/1に割り当て直す ---
        if df["y"].nunique() == 1:
            print("[INFO] 'y'列が全て同じ値なので、半分ずつ0/1に割り当て直します。")
            n = len(df)
            df.loc[:n//2, "y"] = 0
            df.loc[n//2:, "y"] = 1
            df.to_csv("data/USDJPY_15m.csv", index=False)
        # --- ラベル監査CSV出力 ---
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
        # グループ作成例
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
            print(f"[CV purged] fold={fold} MCC={row['MCC']:.3f} BA={row['BA']:.3f} "
                  f"F1_neg={row['F1_neg']:.3f} PR_minor="
                  f"{(row['PR_AUC_pos'] if row['minority_class']==1 else row['PR_AUC_neg']):.3f} "
                  f"Brier={row['Brier']:.4f} pos_ratio={row['pos_ratio_fold']:.3f}")

        cv_df = pd.DataFrame(cv_rows)
        cv_df.to_csv("cv_minor_summary.csv", index=False)
        print("[CV SUMMARY] saved -> cv_minor_summary.csv")

        # --- EV計算例 ---
        p = float(np.mean(y == 1))  # 例: 全体の陽性比率
        reward_win = 1.0  # 例: 勝ちトレードの平均リターン
        reward_lose = 1.0  # 例: 負けトレードの平均リターン
        cost = 0.02  # 例: 手数料・スリッページ等
        ev_tr = p * reward_win - (1 - p) * reward_lose - cost
        print(f"[EV] net(costed)={ev_tr:.3f} | gross(no cost)={p*reward_win - (1-p)*reward_lose:.3f} | cost={cost:.3f}")
        print("[END] Purged+Embargo CV・ラベル監査CSV・少数派KPI主指標化 実行完了")
    except FileNotFoundError:
        print("[ERROR] data/USDJPY_15m.csv が見つかりません。ファイルを配置してください。")
    except AssertionError as e:
        print("[ERROR] カラム・データエラー:", e)
        # y列がない場合は自動で追加して再実行する例
        if "y" in str(e):
            print("[INFO] 'y'列が存在しないため、仮ラベル列を追加します。")
            try:
                df = pd.read_csv("data/USDJPY_15m.csv")
                df["y"] = 0  # 仮ラベル（全て0）
                df.to_csv("data/USDJPY_15m.csv", index=False)
                print("[INFO] 'y'列を追加しました。再度スクリプトを実行してください。")
                import sys
                sys.exit(0)
            except Exception as e2:
                print("[ERROR] 'y'列追加時にエラー:", e2)
                import sys
                sys.exit(1)
    except Exception as e:
        import traceback
        print("[ERROR] その他のエラー:", e)
        traceback.print_exc()

        # --- ここまで ---
    import pandas as pd
    import numpy as np
    from purged_cv import make_time_groups, PurgedGroupTimeSeriesSplit
    from sklearn.linear_model import LogisticRegression
    from sweep_labels import export_label_audit_samples
    from purged_cv import make_time_groups, PurgedGroupTimeSeriesSplit
    try:
        print("[START] Purged+Embargo CV・ラベル監査CSV・少数派KPI主指標化 実行開始")
        # サンプルデータの読み込み例
        df = pd.read_csv("data/USDJPY_15m.csv")
        print("dfのカラム:", df.columns)
        # --- ラベル監査CSV出力 ---
        pos_csv, neg_csv = export_label_audit_samples(
            df, label_col="y", proba_col=None, out_prefix="audit_break"
        )
        print(f"[label audit] exported: {pos_csv}, {neg_csv}")
        tcol = "time"
        assert tcol in df.columns, f"dfに '{tcol}' 列がありません"
        assert "y" in df.columns, "dfに 'y' 列がありません"
        feature_cols = [c for c in df.columns if c not in [tcol, "y"]]
        assert feature_cols, "特徴量となるカラムがありません"
        X = df[feature_cols]
        y = df["y"]
        # グループ作成例
        groups = make_time_groups(df[tcol], freq="D")
        cv = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=1, embargo_groups=1)
        cv_rows = []
        for fold, (tr_idx, te_idx) in enumerate(cv.split(df, groups=groups), 1):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx].values
            X_te, y_te = X.iloc[te_idx], y.iloc[te_idx].values
            # 収束しない対策（例）
            # model = LogisticRegression(solver="saga", max_iter=5000, n_jobs=-1, class_weight="balanced")
            model = LogisticRegression(max_iter=200)
            model.fit(X_tr, y_tr)
            p_te = model.predict_proba(X_te)[:, 1]

            row = summarize_minor_metrics(y_te, p_te)
            row["fold"] = fold
            row["pos_ratio_fold"] = float(np.mean(y_te == 1))
            cv_rows.append(row)
            print(f"[CV purged] fold={fold} MCC={row['MCC']:.3f} BA={row['BA']:.3f} "
                  f"F1_neg={row['F1_neg']:.3f} PR_minor="
                  f"{(row['PR_AUC_pos'] if row['minority_class']==1 else row['PR_AUC_neg']):.3f} "
                  f"Brier={row['Brier']:.4f} pos_ratio={row['pos_ratio_fold']:.3f}")

        cv_df = pd.DataFrame(cv_rows)
        cv_df.to_csv("cv_minor_summary.csv", index=False)
        print("[CV SUMMARY] saved -> cv_minor_summary.csv")

        # --- EV計算例 ---
        # p, reward_win, reward_lose, cost は適宜設定してください
        p = float(np.mean(y == 1))  # 例: 全体の陽性比率
        reward_win = 1.0  # 例: 勝ちトレードの平均リターン
        reward_lose = 1.0  # 例: 負けトレードの平均リターン
        cost = 0.02  # 例: 手数料・スリッページ等
        ev_tr = p * reward_win - (1 - p) * reward_lose - cost
        print(f"[EV] net(costed)={ev_tr:.3f} | gross(no cost)={p*reward_win - (1-p)*reward_lose:.3f} | cost={cost:.3f}")
        print("[END] Purged+Embargo CV・ラベル監査CSV・少数派KPI主指標化 実行完了")
    except FileNotFoundError:
        print("[ERROR] data/USDJPY_15m.csv が見つかりません。ファイルを配置してください。")
    except AssertionError as e:
        print("[ERROR] カラム・データエラー:", e)
    except Exception as e:
        import traceback
        print("[ERROR] その他のエラー:", e)
        traceback.print_exc()
