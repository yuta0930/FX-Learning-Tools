from __future__ import annotations

from pathlib import Path
import os
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


@st.cache_data(show_spinner=False)
def _read_any_cached(path_str: str, mtime: float, nonce: float | None = None) -> pd.DataFrame:
    """Cache-aware reader keyed by file mtime to avoid stale data.

    - Cache invalidates immediately when the file's modified time changes.
    - Keeps Streamlit cache benefits without waiting for a fixed TTL.
    """
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    try:
        mtime = p.stat().st_mtime
    except Exception:
        mtime = 0.0
    # Include a session-level nonce to force refresh even if mtime is unchanged
    try:
        nonce = float(st.session_state.get("atr_force_reload_key")) if "atr_force_reload_key" in st.session_state else None
    except Exception:
        nonce = None
    return _read_any_cached(str(p), mtime, nonce)


def _ensure_time_utc(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    d = df.copy()
    # If desired column exists, normalize it and return
    if time_col in d.columns:
        t = pd.to_datetime(d[time_col], errors="coerce", utc=True)
        d[time_col] = t
        return d
    # Fallbacks: common time column names → create `time` column
    for alt in ["timestamp", "date", "datetime", "Time", "Timestamp", "Date", "Datetime"]:
        if alt in d.columns:
            t = pd.to_datetime(d[alt], errors="coerce", utc=True)
            d[time_col] = t
            return d
    # If index looks like time, also expose a column (avoid double management by not setting index here)
    if isinstance(d.index, pd.DatetimeIndex):
        d[time_col] = pd.to_datetime(d.index, errors="coerce", utc=True)
    return d


def _trim_trailing_invalid(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Drop trailing rows where any of the specified columns are non-finite.

    Returns: (trimmed_df, n_dropped)
    """
    if df is None or len(df) == 0:
        return df, 0
    d = df.copy()
    n_drop = 0
    while len(d) > 0:
        row = d.iloc[-1]
        bad = False
        for c in cols:
            if c not in d.columns:
                bad = True
                break
            v = row[c]
            try:
                v = float(v)
            except Exception:
                v = np.nan
            if not np.isfinite(v):
                bad = True
                break
        if not bad:
            break
        d = d.iloc[:-1]
        n_drop += 1
    return d, n_drop


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

        # 4) If ATR exists but last value is NaN, try to recompute from OHLC
        try:
            need = {"high", "low", "close"}
            has_ohlc = need.issubset({c.lower() for c in data.columns})
            if atr_col in data.columns and has_ohlc:
                last_atr_try = pd.to_numeric(data[atr_col], errors="coerce").astype(float).iloc[-1]
                if not np.isfinite(last_atr_try):
                    # Force recompute: drop existing atr and rebuild
                    from src.core.ta import add_atr_if_missing as _add

                    tmp = data.copy()
                    try:
                        if atr_col in tmp.columns:
                            tmp = tmp.drop(columns=[atr_col])
                    except Exception:
                        pass
                    tmp = _add(tmp)
                    if atr_col in tmp.columns:
                        data[atr_col] = tmp[atr_col]
        except Exception:
            pass

        # 5) Trim trailing incomplete bars for display stability
        # Prefer both price and atr to be finite in the last row
        must_have = [price_col]
        if atr_col in data.columns:
            must_have.append(atr_col)
        data2, dropped = _trim_trailing_invalid(data, must_have)
        if len(data2) == 0:
            st.caption("ATRパネル: 有効な終値/ATRのバーが無いためスキップ（No-Op）。")
            return
        if dropped > 0:
            st.caption(f"末尾の未完成バー {dropped} 本をスキップして表示しています。")

        data = data2

        # 6) Pick last values
        if atr_col not in data.columns or price_col not in data.columns:
            st.caption("ATRパネル: 必要列（atr/close）が不足のためスキップ（No-Op）。")
            return

        last_series = _coerce_float(data[atr_col])
        last_atr = float(last_series.iloc[-1])
        # 前バーのATR（delta表示用）
        prev_atr = float("nan")
        try:
            if len(last_series.dropna()) >= 2:
                prev_atr = float(last_series.dropna().iloc[-2])
        except Exception:
            pass
        last_price = float(_coerce_float(data[price_col]).iloc[-1])
        # 最終バー時刻（視認性向上: 値が変わらない時のヒント）
        last_time_txt = "-"
        try:
            if time_col in data.columns and len(data[time_col]) > 0:
                t_last = pd.to_datetime(data[time_col].iloc[-1], errors="coerce", utc=True)
                if pd.notna(t_last):
                    # ローカルタイムの簡易表示
                    last_time_txt = str(pd.Timestamp(t_last).tz_convert("Asia/Tokyo").strftime("%Y-%m-%d %H:%M")) + " JST"
        except Exception:
            pass
        if not (np.isfinite(last_atr) and np.isfinite(last_price)):
            st.caption("ATRパネル: 値がNaNのためスキップ（No-Op）。")
            return

    # 7) Decide session and load targets
        session_name = _get_session(data, time_col=time_col, session_col=session_col)
        cfg = _load_atr_targets_yaml()
        tgt = _pick_targets(session_name, cfg)
        k_sl = float(tgt.get("k_sl", 0.60))
        k_tp = float(tgt.get("k_tp", 1.50))
        timeout_bars = int(tgt.get("timeout_bars", 10))
        min_spread_pips = float(tgt.get("min_spread_pips", 1.0))

    # 8) Compute metrics
        m = _compute_metrics(last_price, last_atr, k_sl=k_sl, k_tp=k_tp, pip_size=float(pip_size))
        atr_pips = last_atr / float(pip_size)
        delta_pips = float("nan")
        if np.isfinite(prev_atr):
            delta_pips = (last_atr - prev_atr) / float(pip_size)

        # Spread safety (optional)
        spread_safe_txt = "-"
        if spread_col in data.columns:
            sp = float(_coerce_float(data[spread_col]).iloc[-1])
            # Require SL >= max(2*spread, min_spread_pips)
            thr = max(2.0 * sp, float(min_spread_pips))
            ok = m["sl_pips"] >= thr if np.isfinite(m["sl_pips"]) else False
            spread_safe_txt = f"OK (SL≥{thr:.1f}p)" if ok else f"NG (SL<{thr:.1f}p)"

    # 9) Render badges
        c1, c2, c3, c4, c5, c6 = st.columns([1.2, 0.9, 0.9, 1.2, 1.6, 1.0])
        with c1:
            delta_txt = (f"{delta_pips:+.2f}p" if np.isfinite(delta_pips) and abs(delta_pips) >= 0 else None)
            st.metric(
                "ATR(15m)",
                f"{_format_pips(atr_pips)}p",
                delta=delta_txt,
                help=f"period=14 (Wilder/EWM) | last={last_atr:.6f} prev={prev_atr if np.isfinite(prev_atr) else 'N/A'}"
            )
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

        # 最終バーの時刻と手動更新
        c_time, c_btn = st.columns([1.2, 0.8])
        with c_time:
            # データソースのmtimeも参考表示（“昔のまま”の調査に有用）
            src_txt = ""
            abs_path = None
            try:
                if data_path:
                    p = Path(data_path)
                    if p.exists():
                        abs_path = str(p.resolve())
                        mtime = p.stat().st_mtime
                        mtxt = pd.to_datetime(mtime, unit="s").tz_localize("UTC").tz_convert("Asia/Tokyo").strftime("%Y-%m-%d %H:%M")
                        src_txt = f" | ソース: {p.name} (更新 {mtxt} JST)"
            except Exception:
                pass
            st.caption(f"最終バー: {last_time_txt}{src_txt}")
            # 古いデータソースの注意喚起
            try:
                now_utc = pd.Timestamp.utcnow()
                if 't_last' in locals() and pd.notna(t_last) and (now_utc - pd.Timestamp(t_last)).total_seconds() > 24*3600:
                    st.warning("データソースが24時間以上更新されていません。データ生成側またはパス設定をご確認ください。")
            except Exception:
                pass
        with c_btn:
            # ワンクリックで「データ更新 + 再読込」を実行（既定: JPY=X 15m 30d）
            if st.button("更新＋再読込 (JPY=X / 15m / 30d)"):
                try:
                    if not data_path:
                        st.warning("ファイルパス未指定のため更新できません。data_path を指定してください。")
                    else:
                        p = Path(str(data_path))
                        p.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            from scripts.update_price_csv import _download as _dl, _merge_existing as _merge
                        except Exception:
                            _dl = None
                            _merge = None
                        if _dl is None or _merge is None:
                            st.error("アップデータの読み込みに失敗しました。")
                        else:
                            new_df = _dl("JPY=X", "15m", "30d")
                            merged = _merge(p, new_df)
                            if "timestamp" in merged.columns:
                                merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
                            merged.to_csv(p, index=False)
                            import time as _time
                            st.cache_data.clear()
                            st.session_state["atr_force_reload_key"] = _time.time()
                            st.rerun()
                except Exception as e:
                    st.error(f"更新に失敗しました: {e}")

        # 追加UI（診断/個別更新）のプルダウンは簡素化のため撤去済み。

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
