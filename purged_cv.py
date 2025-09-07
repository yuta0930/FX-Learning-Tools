import numpy as np
import pandas as pd
from typing import List, Tuple

def purged_walk_forward_indices(n: int, n_splits: int, embargo: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    n を (n_splits+1) 等分し、各スプリットで「過去→未来」を厳守。
    学習末尾〜検証先頭の間に embargo バーの緩衝を挿入。
    """
    if n_splits < 2:
        raise ValueError("n_splits>=2 を推奨")
    test_size = n // (n_splits + 1)
    idx = np.arange(n)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_splits + 1):
        t0 = k * test_size
        t1 = min(n, t0 + test_size)
        tr_end = max(0, t0 - embargo)
        tr_idx, te_idx = idx[:tr_end], idx[t0:t1]
        if len(tr_idx) and len(te_idx):
            splits.append((tr_idx, te_idx))
    if not splits:
        raise RuntimeError("データ不足でWF不可（splits/embargo/期間を見直してください）")
    return splits
# === tkinter利用時の安全ガード例 ===
import sys
if "tkinter" in sys.modules:
    import threading
    import tkinter as tk
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("tkinterのGUI操作は必ずメインスレッドで実行してください")
# === サンプル実行用: エラー原因となる箇所を事前点検 ===
def make_time_groups(dt: pd.Series, freq: str = "D") -> np.ndarray:
    """
    tz-aware/tz-naive どちらも許容。
    freq: "D"=日, "W"=週（週は月曜開始に合わせる場合は 'W-MON' 等）
    """
    s = pd.to_datetime(dt)
    # 1) 週単位なら floor('W-MON') などで切る
    if freq.upper().startswith("W"):
        key = s.dt.tz_convert("UTC") if s.dt.tz is not None else s
        key = key.dt.floor(freq)  # "W-MON" 等も可
    else:
        # 日単位は normalize でOK（tz-awareでも日境界でそろう）
        key = s.dt.tz_convert("UTC") if s.dt.tz is not None else s
        key = key.dt.normalize()

    codes, _ = pd.factorize(key, sort=True)
    return codes

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Lopez de Prado 流の purge + embargo を実装した GroupTimeSeriesSplit
    - グループ（例：日/週）単位で時系列順に分割
    - 学習・検証の境界付近を purge
    - 検証直後を embargo（情報リーク防止）
    """
    def __init__(self, n_splits: int = 5, group_gap: int = 0, embargo_groups: int = 1):
        self.n_splits = n_splits
        self.group_gap = group_gap
        self.embargo_groups = embargo_groups

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups（グループID配列）が必要です")
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        n_groups = len(uniq)
        if self.n_splits >= n_groups:
            raise ValueError("n_splits はグループ数未満にしてください")

        fold_sizes = np.full(self.n_splits, n_groups // self.n_splits, dtype=int)
        fold_sizes[: n_groups % self.n_splits] += 1
        current = 0
        boundaries = []
        for fs in fold_sizes:
            boundaries.append((current, current + fs))
            current += fs

        # 時系列順に各foldの検証グループを割当
        for i, (start, stop) in enumerate(boundaries):
            test_groups = uniq[start:stop]
            # purge: 学習の終端と検証の始端の間に group_gap を空ける
            left = start - self.group_gap
            train_left = max(0, left)
            # embargo: 検証の直後 embargo_groups を学習から除外
            right = stop + self.embargo_groups
            train_groups = np.concatenate([uniq[:train_left], uniq[right:]])
            # インデックス抽出
            train_idx = np.where(np.isin(groups, train_groups))[0]
            test_idx  = np.where(np.isin(groups,  test_groups))[0]
            yield train_idx, test_idx

if __name__ == "__main__":
    import pandas as pd
    try:
        # サンプルデータの読み込み例
        df = pd.read_csv("data/USDJPY_15m.csv")
        print("dfのカラム:", df.columns)
        tcol = "time"
        assert tcol in df.columns, f"dfに '{tcol}' 列がありません"
        # X, y の例（特徴量と目的変数）
        # 必要に応じてカラム名を調整してください
        feature_cols = [c for c in df.columns if c not in [tcol, "y"]]
        assert feature_cols, "特徴量となるカラムがありません"
        assert "y" in df.columns, "dfに 'y' 列がありません"
        X = df[feature_cols]
        y = df["y"]
        # グループ作成
        groups = make_time_groups(df[tcol], freq="D")
        assert len(groups) == len(df), "groupsの長さがdfと一致しません"
        cv = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=1, embargo_groups=1)
        for fold, (tr_idx, te_idx) in enumerate(cv.split(df, groups=groups), 1):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
            print(f"fold{fold}: train={len(tr_idx)}, test={len(te_idx)}")
            # ここで学習・予測処理を記述
    except FileNotFoundError:
        print("data/USDJPY_15m.csv が見つかりません。ファイルを配置してください。")
    except AssertionError as e:
        print("カラム・データエラー:", e)
    except Exception as e:
        print("その他のエラー:", e)
