import numpy as np
from sklearn.metrics import brier_score_loss

# ダミー: 校正の効果を簡易確認（人工データでPlattによる改善）

def sigmoid(x):
    return 1/(1+np.exp(-x))

def test_calibration_brier_improves_on_synthetic():
    rng = np.random.default_rng(42)
    n=1000
    # 真の確率
    p_true = sigmoid(rng.normal(0,1,n))
    y = rng.binomial(1, p_true)
    # 過信モデル（スケール2倍）
    logit_over = np.log(p_true/(1-p_true))*2.0
    p_over = 1/(1+np.exp(-logit_over))
    # 単純温度スケーリング T>1 で改善する想定
    T=1.8
    logit_adj = logit_over / T
    p_adj = 1/(1+np.exp(-logit_adj))
    b_over = brier_score_loss(y, p_over)
    b_adj = brier_score_loss(y, p_adj)
    assert b_adj < b_over
