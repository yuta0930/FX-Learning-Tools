import numpy as np
from ev_utils import EVConfig
from ai_train_break import search_best_theta


def test_search_best_theta_basic():
    rng = np.random.default_rng(0)
    proba = rng.uniform(0.4, 0.9, 500)
    ev = EVConfig(R_win=1.0, R_loss=1.0, cost_per_trade=0.05)
    res = search_best_theta(proba, ev, min_cov=0.05, target_cov=0.10)
    assert 'theta' in res and 'ev_per_trade' in res
    assert 0.5 <= res['theta'] <= 0.95
    assert res['coverage'] >= 0.0


def test_search_best_theta_no_target_cov_reached():
    # All probabilities low -> target_cov path likely fails
    proba = np.linspace(0.50, 0.60, 200)
    ev = EVConfig(R_win=1.0, R_loss=1.0, cost_per_trade=0.2)
    res = search_best_theta(proba, ev, min_cov=0.05, target_cov=0.40)  # unreachable target
    # Should fallback to min_cov or trades>0 branch
    assert res['trades'] >= 0
    assert 'ev_per_trade' in res
