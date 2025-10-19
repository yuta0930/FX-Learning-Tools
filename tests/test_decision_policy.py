import math
from decision_policy import recommend_action, EVConfig, DecisionParams


def _base_params(**kw):
    p = DecisionParams(
        min_ev_r=0.0,
        theta_bump_by_regime={"mid": 0.02},
        theta_bump_in_news=0.03,
        news_mode="soft",
        spread_max=0.03,
        wick_ratio_max=2.0,
        prefer_limit_retest=False,
        rationale_verbose=False,
        pattern_ev_weight=0.2,
    )
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_news_hard_blocks_unconditionally():
    params = _base_params(news_mode="hard")
    ev = EVConfig()
    act, reasons, th_eff, evr = recommend_action(
        p=0.9, theta=0.6, session="Tokyo", feat_row={}, ohlc=(1,2,0.5,1.5),
        in_news=True, spread=0.01, ev_cfg=ev, params=params
    )
    assert act == "見送り"
    assert any("重要イベント" in r for r in reasons)


def test_soft_news_bumps_theta_and_can_block():
    params = _base_params(news_mode="soft", theta_bump_in_news=0.1)
    ev = EVConfig()
    # p が baseθ未満だが、soft bump後のθ_effに届かない → 見送り
    act, reasons, th_eff, evr = recommend_action(
        p=0.68, theta=0.65, session="Tokyo", feat_row={}, ohlc=(1,2,0.5,1.5),
        in_news=True, spread=0.01, ev_cfg=ev, params=params
    )
    assert act == "見送り"
    assert th_eff >= 0.75  # 0.65 + 0.1 = 0.75


def test_theta_and_ev_gate_allows_with_pattern_bonus():
    params = _base_params(min_ev_r=0.05)
    ev = EVConfig(cost_per_trade=0.0)
    # p が θを満たし、pattern EVのボーナスで EV/R* が min_ev を超える
    act, reasons, th_eff, evr = recommend_action(
        p=0.62, theta=0.6, session="Tokyo", feat_row={"reg_atr_mid": 1.0}, ohlc=(1,2,0.5,1.5),
        in_news=False, spread=0.01, ev_cfg=ev, params=params, pattern_ev_r=0.5
    )
    assert act in ("成行(小)", "指値(リテスト)")
    assert any("EV/R" in r for r in reasons)


def test_spread_blocks():
    params = _base_params(spread_max=0.01)
    ev = EVConfig()
    act, reasons, th_eff, evr = recommend_action(
        p=0.9, theta=0.6, session="Tokyo", feat_row={}, ohlc=(1,2,0.5,1.5),
        in_news=False, spread=0.02, ev_cfg=ev, params=params
    )
    assert act == "見送り"
    assert any("スプレッド" in r for r in reasons)


def test_long_wick_prefers_limit():
    params = _base_params(wick_ratio_max=1.5, prefer_limit_retest=True)
    ev = EVConfig(cost_per_trade=0.0)
    act, reasons, th_eff, evr = recommend_action(
        p=0.9, theta=0.6, session="Tokyo", feat_row={}, ohlc=(1,3.5,0.5,1.6),
        in_news=False, spread=0.005, ev_cfg=ev, params=params
    )
    assert act == "指値(リテスト)"
    assert any("リテスト" in r for r in reasons)
