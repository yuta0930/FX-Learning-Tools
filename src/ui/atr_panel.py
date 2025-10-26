from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, Dict

import pandas as pd
import numpy as np
import streamlit as st


# --- Config loader (optional external, else local) ---
def _load_targets_yaml_local(path: Path = Path("config/atr_targets.yml")) -> Dict[str, Any]:
    """Load ATR target settings from YAML with safe defaults.

    Structure:
    - default: {k_sl, k_tp, timeout_bars, min_spread_pips}
    - sessions: { Tokyo: {...}, London: {...}, NewYork: {...} }
    """
    try:
        import yaml  # type: ignore
    except Exception:
        return {
            "default": {"k_sl": 0.60, "k_tp": 1.50, "timeout_bars": 10, "min_spread_pips": 1.0},
            "sessions": {},
        }

    try:
        if not path.exists():
            return {
                "default": {"k_sl": 0.60, "k_tp": 1.50, "timeout_bars": 10, "min_spread_pips": 1.0},
                "sessions": {},
            }
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("invalid yaml structure")
        default = data.get("default") or {}
        sessions = data.get("sessions") or {}
        defv = {
            "k_sl": float(default.get("k_sl", 0.60)),
            "k_tp": float(default.get("k_tp", 1.50)),
            "timeout_bars": int(default.get("timeout_bars", 10)),
            "min_spread_pips": float(default.get("min_spread_pips", 1.0)),
        }
        # sanitize sessions
        sess_out: Dict[str, Dict[str, Any]] = {}
        if isinstance(sessions, dict):
            for k, v in sessions.items():
                if isinstance(v, dict):
                    sess_out[str(k)] = {
                        "k_sl": float(v.get("k_sl", defv["k_sl"])),
                        "k_tp": float(v.get("k_tp", defv["k_tp"])),
                        "timeout_bars": int(v.get("timeout_bars", defv["timeout_bars"])),
                        "min_spread_pips": float(v.get("min_spread_pips", defv["min_spread_pips"])),
                    }
        return {"default": defv, "sessions": sess_out}
    except Exception:
        return {
            "default": {"k_sl": 0.60, "k_tp": 1.50, "timeout_bars": 10, "min_spread_pips": 1.0},
            "sessions": {},
        }


def _load_atr_targets_yaml() -> Dict[str, Any]:
    """Try external loader src.policy.atr_targets.load_atr_targets_yaml, else local."""
    try:
        from src.policy.atr_targets import load_atr_targets_yaml  # type: ignore

        try:
            return load_atr_targets_yaml()  # type: ignore
        except Exception:
            return _load_targets_yaml_local()
    except Exception:
        return _load_targets_yaml_local()


@st.cache_data(show_spinner=False, ttl=300)
def _read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _ensure_time_utc(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    d = df.copy()
    if time_col in d.columns:
        t = pd.to_datetime(d[time_col], errors="coerce", utc=True)
        d[time_col] = t
        return d
    # If index looks like time, also expose a column (avoid double management by not setting index here)
    if isinstance(d.index, pd.DatetimeIndex):
        d[time_col] = pd.to_datetime(d.index, errors="coerce", utc=True)
    return d


def _coerce_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _format_pips(x: float) -> str:
    if not np.isfinite(x):
        return "-"
    # 1～2桁表示のラフな丸め
    return f"{x:.1f}" if abs(x) >= 1 else f"{x:.2f}"


def _get_session(df: pd.DataFrame, time_col: str = "time", session_col: str = "session") -> str:
    # Prefer existing session column's last value
    if session_col in df.columns and len(df) > 0:
        val = str(df.iloc[-1][session_col])
        if val:
            return val
    # Try official helper
    try:
        from src.core.session import add_session

        dd = add_session(df.copy(), time_col=time_col)
        if session_col in dd.columns and len(dd) > 0:
            return str(dd.iloc[-1][session_col])
    except Exception:
        pass
    # Fallback: UTC hour → rough session
    try:
        t = pd.to_datetime(df.iloc[-1][time_col], errors="coerce", utc=True)
        if pd.isna(t):
            return "Other"
        h = int(pd.Timestamp(t).hour)
        # Rough buckets (example)
        if 0 <= h < 8:
            return "Tokyo"
        if 8 <= h < 16:
            return "London"
        if 16 <= h < 24:
            return "NewYork"
    except Exception:
        pass
    return "Other"


def _pick_targets(session_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = cfg.get("default", {}) if isinstance(cfg, dict) else {}
    sess = (cfg.get("sessions", {}) or {}) if isinstance(cfg, dict) else {}
    if session_name in sess:
        base = dict(d)
        base.update(sess[session_name])
        return base
    return d or {"k_sl": 0.60, "k_tp": 1.50, "timeout_bars": 10, "min_spread_pips": 1.0}


def _compute_metrics(
    last_price: float,
    last_atr: float,
    *,
    k_sl: float,
    k_tp: float,
    pip_size: float,
) -> Dict[str, float]:
    sl_dist_price = last_atr * k_sl
    tp_dist_price = last_atr * k_tp
    sl_pips = sl_dist_price / pip_size
    tp_pips = tp_dist_price / pip_size
    rr = (k_tp / k_sl) if (k_sl and np.isfinite(k_sl) and k_sl != 0) else float("nan")
    return {
        "sl_price": sl_dist_price,
        "tp_price": tp_dist_price,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "rr": rr,
    }


def render_atr_panel(
    df: Optional[pd.DataFrame] = None,
    data_path: Optional[Path | str] = None,
    *,
    atr_col: str = "atr",
    price_col: str = "close",
    time_col: str = "time",
    session_col: str = "session",
    spread_col: str = "spread_pips",
    pip_size: float = 0.01,
) -> None:
    """Render a persistent ATR/targets panel below the page title.

    Safe no-op with explanatory caption when data/config is unavailable.
    """
    try:
        # 1) Resolve DataFrame
        data: Optional[pd.DataFrame] = None
        if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
            data = df.copy()
        elif data_path:
            try:
                data = _read_any(data_path)
            except Exception:
                data = None

        if data is None or len(data) == 0:
            st.caption("ATRパネル: 入力データが無いためスキップ（No-Op）。config/atr_targets.yml があれば設定が反映されます。")
            return

        # 2) Normalize time to UTC column
        data = _ensure_time_utc(data, time_col=time_col)

        # 3) Add ATR if missing
        try:
            from src.core.ta import add_atr_if_missing

            data = add_atr_if_missing(data)
        except Exception:
            pass  # keep as-is

        # 4) Pick last values
        if atr_col not in data.columns or price_col not in data.columns:
            st.caption("ATRパネル: 必要列（atr/close）が不足のためスキップ（No-Op）。")
            return

        last_atr = float(_coerce_float(data[atr_col]).iloc[-1])
        last_price = float(_coerce_float(data[price_col]).iloc[-1])
        if not (np.isfinite(last_atr) and np.isfinite(last_price)):
            st.caption("ATRパネル: 値がNaNのためスキップ（No-Op）。")
            return

        # 5) Decide session and load targets
        session_name = _get_session(data, time_col=time_col, session_col=session_col)
        cfg = _load_atr_targets_yaml()
        tgt = _pick_targets(session_name, cfg)
        k_sl = float(tgt.get("k_sl", 0.60))
        k_tp = float(tgt.get("k_tp", 1.50))
        timeout_bars = int(tgt.get("timeout_bars", 10))
        min_spread_pips = float(tgt.get("min_spread_pips", 1.0))

        # 6) Compute metrics
        m = _compute_metrics(last_price, last_atr, k_sl=k_sl, k_tp=k_tp, pip_size=float(pip_size))
        atr_pips = last_atr / float(pip_size)

        # Spread safety (optional)
        spread_safe_txt = "-"
        if spread_col in data.columns:
            sp = float(_coerce_float(data[spread_col]).iloc[-1])
            # Require SL >= max(2*spread, min_spread_pips)
            thr = max(2.0 * sp, float(min_spread_pips))
            ok = m["sl_pips"] >= thr if np.isfinite(m["sl_pips"]) else False
            spread_safe_txt = f"OK (SL≥{thr:.1f}p)" if ok else f"NG (SL<{thr:.1f}p)"

        # 7) Render badges
        c1, c2, c3, c4, c5, c6 = st.columns([1.2, 0.9, 0.9, 1.2, 1.6, 1.0])
        with c1:
            st.metric("ATR(15m)", f"{_format_pips(atr_pips)}p", help=f"price={last_atr:.5f}")
        with c2:
            st.metric("kSL", f"{k_sl:.2f}")
        with c3:
            st.metric("kTP", f"{k_tp:.2f}")
        with c4:
            st.metric("RR (kTP/kSL)", f"{m['rr']:.2f}")
        with c5:
            st.metric(
                "SL/TP 距離",
                f"SL { _format_pips(m['sl_pips']) }p / TP { _format_pips(m['tp_pips']) }p",
                help=f"SL={m['sl_price']:.5f} | TP={m['tp_price']:.5f}",
            )
        with c6:
            st.metric("⏱ timeout", f"{timeout_bars} bars")

        # Spread guard indicator (if available)
        if spread_col in data.columns:
            if "NG" in spread_safe_txt:
                st.error(f"Spreadガード: {spread_safe_txt}")
            else:
                st.success(f"Spreadガード: {spread_safe_txt}")
        else:
            st.caption("spread_pips 列が無いため、Spreadガードはスキップ（任意）。")

    except Exception as e:
        # Never crash the page; show minimal info only
        st.caption(f"ATRパネルの表示をスキップしました（No-Op）: {e}")
