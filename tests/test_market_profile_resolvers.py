import pytest

from src.core.market_profile import (
    resolve_theta_with_profile,
    resolve_atr_filter_with_profile,
    resolve_risk_guard_overrides,
)


AUTO_CFG = {
    "profiles": {
        "Tokyo": {
            "mid": {
                "theta_offset": 0.05,
                "atr_filter": {"min_atr": 1.1, "max_atr": 2.2},
                "risk_guard": {"max_trades_per_day": 5},
            }
        },
        "__default__": {
            "mid": {
                "theta_offset": -0.02,
                "atr_filter": {"min_atr": 0.7, "max_atr": 1.8},
                "risk_guard": {"max_trades_per_day": 15},
            }
        },
    }
}


def test_resolve_theta_with_profile_applies_offset():
    result = resolve_theta_with_profile(
        0.60,
        session="Tokyo",
        atr_regime="mid",
        cfg_auto=AUTO_CFG,
    )
    assert result == pytest.approx(0.65)


def test_resolve_atr_filter_with_profile_overrides_defaults():
    min_atr, max_atr = resolve_atr_filter_with_profile(
        default_min=0.5,
        default_max=3.0,
        session="Tokyo",
        atr_regime="mid",
        cfg_auto=AUTO_CFG,
    )
    assert min_atr == pytest.approx(1.1)
    assert max_atr == pytest.approx(2.2)


def test_resolve_risk_guard_overrides_handles_fallback():
    overrides_tokyo = resolve_risk_guard_overrides(
        session="Tokyo",
        atr_regime="mid",
        cfg_auto=AUTO_CFG,
    )
    overrides_fallback = resolve_risk_guard_overrides(
        session="London",
        atr_regime="mid",
        cfg_auto=AUTO_CFG,
    )

    assert overrides_tokyo["max_trades_per_day"] == 5
    assert overrides_fallback["max_trades_per_day"] == 15
