from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple
import math

import pandas as pd

from src.core.ta import add_atr_if_missing

ATR_LOW = "low"
ATR_MID = "mid"
ATR_HIGH = "high"
_VALID_REGIMES = {ATR_LOW, ATR_MID, ATR_HIGH}

_SESSION_ALIASES = {
    "tokyo": "Tokyo",
    "london": "London",
    "ny": "NewYork",
    "newyork": "NewYork",
    "nyc": "NewYork",
}


def _as_plain_dict(node: Any) -> Mapping[str, Any]:
    if node is None:
        return {}
    if isinstance(node, Mapping):
        return node
    if hasattr(node, "to_dict"):
        try:
            data = node.to_dict()
            if isinstance(data, Mapping):
                return data
        except Exception:
            return {}
    return {}


def _canonical_session(session: Optional[str]) -> Optional[str]:
    if session is None:
        return None
    key = str(session).strip()
    if not key:
        return None
    lowered = key.lower()
    if lowered in _SESSION_ALIASES:
        return _SESSION_ALIASES[lowered]
    return key


def _profile_for(cfg_auto: Mapping[str, Any], session: Optional[str], atr_regime: str) -> Mapping[str, Any]:
    profiles = _as_plain_dict(cfg_auto.get("profiles"))
    if not profiles:
        return {}
    session_key = _canonical_session(session)
    session_dict = _as_plain_dict(profiles.get(session_key)) if session_key in profiles else {}
    if not session_dict:
        session_dict = _as_plain_dict(profiles.get("__default__"))
    if not session_dict:
        return {}
    regime_key = atr_regime if atr_regime in _VALID_REGIMES else ATR_MID
    regime_dict = _as_plain_dict(session_dict.get(regime_key))
    if not regime_dict and regime_key != ATR_MID:
        regime_dict = _as_plain_dict(session_dict.get(ATR_MID))
    if not regime_dict:
        default_profile = _as_plain_dict(profiles.get("__default__"))
        if default_profile:
            regime_dict = _as_plain_dict(default_profile.get(regime_key))
            if not regime_dict and regime_key != ATR_MID:
                regime_dict = _as_plain_dict(default_profile.get(ATR_MID))
    return regime_dict


def _pick_timestamp(price_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    for col in ("timestamp", "time", "datetime"):
        if col in price_df.columns:
            ts = pd.to_datetime(price_df[col], errors="coerce")
            valid = ts.dropna()
            if not valid.empty:
                return pd.Timestamp(valid.iloc[-1])
    if isinstance(price_df.index, pd.DatetimeIndex) and len(price_df.index) > 0:
        return pd.Timestamp(price_df.index[-1])
    return None


def _pick_session(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None:
        return None
    try:
        h = int(ts.hour)
    except Exception:
        return None
    if 9 <= h < 15:
        return "Tokyo"
    if 16 <= h < 24:
        return "London"
    if h >= 22 or h < 5:
        return "NewYork"
    return None


@dataclass
class MarketProfile:
    session: Optional[str]
    atr_regime: str
    latest_atr: Optional[float] = None
    time_bucket: Optional[str] = None


def classify_atr_regime(
    atr_series: pd.Series,
    cfg_auto: Any,
) -> Tuple[float, str]:
    series = pd.to_numeric(atr_series, errors="coerce").dropna()
    latest = float(series.iloc[-1]) if not series.empty else math.nan
    cfg_dict = _as_plain_dict(cfg_auto)
    regime_cfg = _as_plain_dict(cfg_dict.get("atr_regime"))
    lookback = int(regime_cfg.get("lookback_bars", 96) or 0)
    min_samples = int(regime_cfg.get("min_samples", 50) or 0)
    low_q = float(regime_cfg.get("low_quantile", 0.3) or 0.3)
    high_q = float(regime_cfg.get("high_quantile", 0.7) or 0.7)

    if lookback > 0 and len(series) > lookback:
        series = series.iloc[-lookback:]

    if len(series) < max(1, min_samples):
        return latest, ATR_MID

    try:
        low_thr = series.quantile(low_q)
        high_thr = series.quantile(high_q)
    except Exception:
        return latest, ATR_MID

    if latest <= low_thr:
        return latest, ATR_LOW
    if latest >= high_thr:
        return latest, ATR_HIGH
    return latest, ATR_MID


def build_market_profile(
    price_df: pd.DataFrame,
    cfg: Any,
    *,
    now_ts: Optional[pd.Timestamp] = None,
    atr_column: str = "atr",
    atr_period: int = 14,
) -> MarketProfile:
    if price_df is None or len(price_df) == 0:
        return MarketProfile(session=None, atr_regime=ATR_MID, latest_atr=None)

    df = price_df
    if atr_column not in df.columns:
        df = add_atr_if_missing(df, period=atr_period, out_col=atr_column)
    atr_series = pd.to_numeric(df.get(atr_column, pd.Series([], dtype=float)), errors="coerce")

    ts = now_ts or _pick_timestamp(df)
    session = _pick_session(ts)

    cfg_auto = getattr(cfg, "auto_params", None)
    cfg_auto_dict = _as_plain_dict(cfg_auto)
    latest_atr, regime = classify_atr_regime(atr_series, cfg_auto_dict)
    return MarketProfile(session=session, atr_regime=regime, latest_atr=latest_atr)


def resolve_theta_with_profile(
    base_theta: float,
    *,
    session: Optional[str],
    atr_regime: str,
    cfg_auto: Any,
) -> float:
    profile = _profile_for(_as_plain_dict(cfg_auto), session, atr_regime)
    offset = profile.get("theta_offset", 0.0)
    try:
        offset_val = float(offset) if offset is not None else 0.0
    except Exception:
        offset_val = 0.0
    return float(base_theta) + offset_val


def resolve_atr_filter_with_profile(
    *,
    default_min: Optional[float],
    default_max: Optional[float],
    session: Optional[str],
    atr_regime: str,
    cfg_auto: Any,
) -> Tuple[Optional[float], Optional[float]]:
    profile = _profile_for(_as_plain_dict(cfg_auto), session, atr_regime)
    atr_filter = _as_plain_dict(profile.get("atr_filter")) if profile else {}
    min_override = atr_filter.get("min_atr")
    max_override = atr_filter.get("max_atr")
    min_val = _coalesce_override(min_override, default_min)
    max_val = _coalesce_override(max_override, default_max)
    return min_val, max_val


def resolve_risk_guard_overrides(
    *,
    session: Optional[str],
    atr_regime: str,
    cfg_auto: Any,
) -> Mapping[str, Any]:
    profile = _profile_for(_as_plain_dict(cfg_auto), session, atr_regime)
    return _as_plain_dict(profile.get("risk_guard")) if profile else {}


def _coalesce_override(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default
