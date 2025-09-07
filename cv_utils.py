import numpy as np
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
