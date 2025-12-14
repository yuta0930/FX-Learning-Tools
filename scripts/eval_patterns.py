from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from src.core.ta import add_atr_if_missing
from src.patterns.flag_pennant import detect_flag_pennant_simplified, FlagPennantSignal

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name is None or not np.issubdtype(df.index.dtype, np.datetime64):
        # Try common column names
        for col in ["time", "timestamp", "datetime", "date"]:
            if col in df.columns:
                out = df.copy()
                out.index = pd.to_datetime(out[col])
                return out
        # Fallback: set monotonic integer index
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return out
    return df


def _add_session(df: pd.DataFrame, time_col: Optional[str] = None) -> pd.DataFrame:
    # Simple UTC-based session bucketing
    def _session_of(ts: pd.Timestamp) -> str:
        h = ts.hour
        # Tokyo: 0-8 UTC, London: 7-16 UTC, NewYork: 12-21 UTC
        if 0 <= h <= 8:
            return "Tokyo"
        if 7 <= h <= 16:
            return "London"
        if 12 <= h <= 21:
            return "NewYork"
        return "Other"

    out = df.copy()
    if time_col and time_col in out.columns:
        tser = pd.to_datetime(out[time_col])
    else:
        tser = pd.to_datetime(out.index)
        out["time"] = tser
    out["session"] = tser.map(_session_of)
    return out


def _load_yaml(path: Path) -> dict:
    if yaml is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data or {}
    except Exception:
        return {}


def _load_events_windows(path: Optional[Path], before_min: int, after_min: int):
    if not path or not path.exists():
        return []
    try:
        ev = pd.read_csv(path)
    except Exception:
        return []
    if "time" not in ev.columns:
        return []
    t = pd.to_datetime(ev["time"])
    return [(tt - pd.Timedelta(minutes=before_min), tt + pd.Timedelta(minutes=after_min)) for tt in t]


def _first_hit_idx(
    highs: np.ndarray,
    lows: np.ndarray,
    entry: float,
    up_level: float,
    dn_level: float,
    max_ahead: int,
) -> Tuple[int | None, str | None]:
    """Return (offset, which) for the first barrier hit within max_ahead bars.
    which ∈ {"up","down"} or (None, None) if none.
    """
    for k in range(1, max_ahead + 1):
        if k >= len(highs):
            break
        if highs[k] >= up_level:
            return k, "up"
        if lows[k] <= dn_level:
            return k, "down"
    return None, None


def _compute_baseline_rates(df: pd.DataFrame, H: int, delta_mult: float, pip_size: float = 0.01) -> Dict[str, float]:
    highs = pd.to_numeric(df["high"], errors="coerce").astype(float).values
    lows = pd.to_numeric(df["low"], errors="coerce").astype(float).values
    closes = pd.to_numeric(df["close"], errors="coerce").astype(float).values
    atr = pd.to_numeric(df["atr"], errors="coerce").astype(float).values
    n = len(df)

    up_first = 0
    dn_first = 0
    denom = 0
    for i in range(n - H - 1):
        if not np.isfinite(atr[i]) or atr[i] <= 1e-12:
            continue
        delta = delta_mult * atr[i]
        up = closes[i] + delta
        dn = closes[i] - delta
        hi_seg = highs[i : i + H + 1]
        lo_seg = lows[i : i + H + 1]
        k, which = _first_hit_idx(hi_seg, lo_seg, closes[i], up, dn, H)
        if which is None:
            denom += 1
            continue
        denom += 1
        if which == "up":
            up_first += 1
        else:
            dn_first += 1
    if denom == 0:
        return {"up_rate": float("nan"), "down_rate": float("nan"), "either_rate": float("nan")}
    return {
        "up_rate": up_first / denom,
        "down_rate": dn_first / denom,
        "either_rate": (up_first + dn_first) / denom,
    }


def _eval_signals(
    df: pd.DataFrame,
    signals: List[FlagPennantSignal],
    H: int,
    delta_mult: float,
    spread_pips: float,
    pip_size: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    highs = pd.to_numeric(df["high"], errors="coerce").astype(float).values
    lows = pd.to_numeric(df["low"], errors="coerce").astype(float).values
    closes = pd.to_numeric(df["close"], errors="coerce").astype(float).values
    atr = pd.to_numeric(df["atr"], errors="coerce").astype(float).values
    pv = pip_size

    rows = []
    for s in signals:
        i = int(s.index)
        if i + 1 >= len(df):
            continue
        if not np.isfinite(atr[i]) or atr[i] <= 1e-12:
            continue
        delta = delta_mult * atr[i]
        up = closes[i] + delta
        dn = closes[i] - delta
        hi_seg = highs[i : i + H + 1]
        lo_seg = lows[i : i + H + 1]
        k, which = _first_hit_idx(hi_seg, lo_seg, closes[i], up, dn, H)
        hit = False
        r_R = 0.0
        if which is not None:
            # Barrier hit in some direction
            if (s.side > 0 and which == "up") or (s.side < 0 and which == "down"):
                hit = True
                r_R = 1.0
            else:
                hit = False
                r_R = -1.0
        else:
            # Time-out: partial PnL normalized by delta
            end = min(len(df) - 1, i + H)
            ret = (closes[end] - closes[i]) if s.side > 0 else (closes[i] - closes[end])
            r_R = float(ret / delta)

        cost_R = (spread_pips * pv) / delta
        r_net = r_R - cost_R

        rows.append(
            {
                "time": df.index[i],
                "index": i,
                "side": s.side,
                "entry": float(closes[i]),
                "delta": float(delta),
                "H": H,
                "hit": bool(hit),
                "r_R": float(r_R),
                "r_net": float(r_net),
                "kind": s.kind,
                "quality": float(s.quality),
            }
        )

    out_df = pd.DataFrame(rows)
    metrics = {
        "n_signals": int(len(out_df)),
        "hit_rate": float(out_df["hit"].mean()) if len(out_df) else float("nan"),
        "ev_R": float(out_df["r_net"].mean()) if len(out_df) else float("nan"),
    }
    return out_df, metrics


def run(
    data_path: str,
    out_dir: str | None,
    H: int,
    delta_mult: float,
    spread_pips: float,
    pip_size: float,
    # Filters
    max_spread_pips: Optional[float] = None,
    events_csv: Optional[str] = None,
    news_minutes_before: int = 30,
    news_minutes_after: int = 30,
    sessions_allow: Optional[str] = None,
    ma_window: Optional[int] = None,
    atr_pctl_min: Optional[float] = None,
    atr_pctl_max: Optional[float] = None,
    # HTF trend alignment (optional)
    htf_enabled: bool = False,
    htf_timeframe: str = "1H",
    htf_ma_len: int = 50,
    htf_slope_window: int = 3,
    # Configs
    patterns_yml: str = "config/patterns.yml",
    patterns_session_yml: str = "config/patterns_session.yml",
    # EV curve constraint
    evcurve_min_n: int = 100,
) -> Tuple[Path, Dict[str, float]]:
    df = pd.read_csv(data_path)
    df = _ensure_time_index(df)
    # Normalize columns
    need = {"open", "high", "low", "close"}
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    if not need.issubset(df.columns):
        raise ValueError(f"dataframe must contain {need}, got {set(df.columns)}")

    df = add_atr_if_missing(df)
    df = _add_session(df)

    # Load defaults for detector
    pat_cfg = _load_yaml(Path(patterns_yml))
    fp = (pat_cfg or {}).get("flag_pennant", {})
    det_kwargs = dict(
        lookback=int(fp.get("lookback", 2000)),
        n_push=int(fp.get("n_push", 30)),
        min_flag_bars=int(fp.get("min_flag_bars", 8)),
        max_flag_bars=int(fp.get("max_flag_bars", 40)),
        sigma_k=float(fp.get("sigma_k", 1.0)),
        pole_min_atr=float(fp.get("pole_min_atr", 2.0)),
        flag_slope_max_atr=float(fp.get("flag_slope_max_atr", 0.15)),
        contraction_percentile=float(fp.get("contraction_percentile", 0.2)),
    )

    # Detect signals
    signals = detect_flag_pennant_simplified(
        df,
        **det_kwargs,
    )

    # Prepare signal DataFrame for filtering/overrides
    sig_rows = [
        {
            "time": s.t0,
            "index": int(s.index),
            "side": int(s.side),
            "entry": float(s.entry),
            "delta": float(s.delta),
            "quality": float(s.quality),
            "kind": s.kind,
            "slope_abs_atr": (float(s.slope_abs_atr) if s.slope_abs_atr is not None else np.nan),
            "contraction_pct": (float(s.contraction_pct) if s.contraction_pct is not None else np.nan),
        }
        for s in signals
    ]
    sig_df = pd.DataFrame(sig_rows)

    # Join market info
    # Build a joinable market-info frame without colliding with existing time/timestamp columns
    df_info = df.copy()
    # Ensure a canonical 'time' column from the index without creating duplicate named columns
    df_info["time"] = pd.to_datetime(df.index)
    df_info = df_info.reset_index(drop=True)
    use_cols = [c for c in ["time", "close", "atr", "spread_pips", "session"] if c in df_info.columns]
    sig_df = sig_df.merge(df_info[use_cols], on="time", how="left")

    # A) Spread filter
    if max_spread_pips is not None and "spread_pips" in sig_df.columns:
        sig_df = sig_df[sig_df["spread_pips"].fillna(0) <= float(max_spread_pips)].copy()

    # A) News window filter
    ev_windows = _load_events_windows(Path(events_csv) if events_csv else None, news_minutes_before, news_minutes_after)
    if ev_windows:
        tvals = pd.to_datetime(sig_df["time"]).to_numpy()
        keep = []
        for tt in tvals:
            ok = True
            for a, b in ev_windows:
                if a <= tt <= b:
                    ok = False
                    break
            keep.append(ok)
        sig_df = sig_df[np.array(keep)].copy()

    # A) Session allow-list
    if sessions_allow:
        allow = {s.strip() for s in str(sessions_allow).split(",") if s.strip()}
        if "session" in sig_df.columns and allow:
            sig_df = sig_df[sig_df["session"].isin(allow)].copy()

    # C) Session-specific threshold overrides (post-detection safe filter)
    sess_cfg = _load_yaml(Path(patterns_session_yml))
    if sess_cfg:
        def _ok_row(row):
            sess = str(row.get("session", ""))
            ov = (sess_cfg or {}).get(sess, {})
            slope_max = float(ov.get("flag_slope_max_atr", det_kwargs["flag_slope_max_atr"]))
            # contraction_pct <= 1 means inside target percentile; allow slight epsilon
            contr_ok = float(row.get("contraction_pct", np.inf)) <= (1.0 + 1e-9)
            slope_ok = float(row.get("slope_abs_atr", np.inf)) <= slope_max
            return bool(slope_ok and contr_ok)

        sig_df = sig_df[sig_df.apply(_ok_row, axis=1)].copy()

    # D) Optional trend filter (MA) and ATR regime filter (No-Op by default)
    if ma_window and ma_window > 1:
        # compute MA over close on the market info df
        if "close" in df.columns:
            ma = pd.Series(pd.to_numeric(df["close"], errors="coerce")).rolling(int(ma_window)).mean()
            ma_df = pd.DataFrame({"time": pd.to_datetime(df.index), "ma": ma.values, "close": df["close"].values})
            sig_df = sig_df.merge(ma_df, on="time", how="left", suffixes=("", ""))
            def _trend_ok(row):
                side = int(row.get("side", 0))
                c = float(row.get("close", np.nan))
                m = float(row.get("ma", np.nan))
                if not np.isfinite(c) or not np.isfinite(m):
                    return False
                return (side > 0 and c >= m) or (side < 0 and c <= m)
            sig_df = sig_df[sig_df.apply(_trend_ok, axis=1)].copy()

    if (atr_pctl_min is not None) or (atr_pctl_max is not None):
        if "atr" in df.columns:
            atr_series = pd.to_numeric(df["atr"], errors="coerce").astype(float)
            lo = np.nanpercentile(atr_series.values, float(atr_pctl_min)) if atr_pctl_min is not None else -np.inf
            hi = np.nanpercentile(atr_series.values, float(atr_pctl_max)) if atr_pctl_max is not None else np.inf
            atr_df = pd.DataFrame({"time": pd.to_datetime(df.index), "atr": atr_series.values})
            sig_df = sig_df.merge(atr_df, on="time", how="left", suffixes=("", ""))
            sig_df = sig_df[(sig_df["atr"].fillna(np.nan) >= lo) & (sig_df["atr"].fillna(np.nan) <= hi)].copy()

    # Baseline evaluation (before HTF filter)
    filt_idx_base = set(int(i) for i in sig_df["index"].tolist())
    signals_base = [s for s in signals if int(s.index) in filt_idx_base]
    sig_eval_df_base, sig_metrics_base = _eval_signals(df, signals_base, H, delta_mult, spread_pips, pip_size)

    # Apply HTF trend alignment if enabled
    sig_df_final = sig_df.copy()
    if htf_enabled and len(sig_df_final):
        # Build HTF series from close
        df_close = pd.DataFrame({"time": pd.to_datetime(df.index), "close": pd.to_numeric(df["close"], errors="coerce").astype(float)})
        df_close = df_close.set_index("time").sort_index()
        df_htf = df_close["close"].resample(htf_timeframe).last().to_frame("htf_close")
        df_htf["htf_ma"] = df_htf["htf_close"].rolling(int(htf_ma_len)).mean()
        df_htf["htf_ma_slope"] = df_htf["htf_ma"].diff(int(htf_slope_window))
        df_htf = df_htf.reset_index().rename(columns={"time": "time_htf"})

        sig_df_final["time_htf"] = pd.to_datetime(sig_df_final["time"]).dt.floor(htf_timeframe)
        sig_df_final = sig_df_final.merge(df_htf, on="time_htf", how="left")

        def _htf_ok(row) -> bool:
            side = int(row.get("side", 0))
            c = float(row.get("htf_close", np.nan))
            ma = float(row.get("htf_ma", np.nan))
            slope = float(row.get("htf_ma_slope", np.nan))
            if not (np.isfinite(c) and np.isfinite(ma) and np.isfinite(slope)):
                return False
            if side > 0:
                return (c > ma) and (slope > 0)
            if side < 0:
                return (c < ma) and (slope < 0)
            return False

        sig_df_final = sig_df_final[sig_df_final.apply(_htf_ok, axis=1)].copy()

    # Evaluate final set (after HTF if enabled)
    filt_idx = set(int(i) for i in sig_df_final["index"].tolist())
    signals_f = [s for s in signals if int(s.index) in filt_idx]
    sig_eval_df, sig_metrics = _eval_signals(df, signals_f, H, delta_mult, spread_pips, pip_size)
    baseline = _compute_baseline_rates(df, H, delta_mult, pip_size=pip_size)

    # Uplift vs directional baseline
    # Use per-direction baseline depending on signal side
    if len(sig_eval_df):
        pos = sig_eval_df[sig_eval_df["side"] > 0]
        neg = sig_eval_df[sig_eval_df["side"] < 0]
        hit_pos = float(pos["hit"].mean()) if len(pos) else np.nan
        hit_neg = float(neg["hit"].mean()) if len(neg) else np.nan
    else:
        hit_pos = hit_neg = np.nan

    uplift_pos = (hit_pos / baseline["up_rate"]) if (np.isfinite(hit_pos) and baseline["up_rate"] not in (0, np.nan)) else np.nan
    uplift_neg = (hit_neg / baseline["down_rate"]) if (np.isfinite(hit_neg) and baseline["down_rate"] not in (0, np.nan)) else np.nan

    metrics = {
        "H": H,
        "delta_mult": delta_mult,
        "spread_pips": spread_pips,
        "pip_size": pip_size,
        "n_rows": int(len(df)),
        "n_signals": int(len(sig_eval_df)),
        "n_signals_baseline": int(len(sig_eval_df_base)),
        "hit_rate": float(sig_metrics["hit_rate"]),
        "ev_R": float(sig_metrics["ev_R"]),
        "ev_R_baseline": float(sig_metrics_base.get("ev_R", float("nan"))),
        "baseline_up": float(baseline["up_rate"]),
        "baseline_down": float(baseline["down_rate"]),
        "baseline_either": float(baseline["either_rate"]),
        "uplift_long_vs_up": float(uplift_pos) if np.isfinite(uplift_pos) else float("nan"),
        "uplift_short_vs_down": float(uplift_neg) if np.isfinite(uplift_neg) else float("nan"),
    }

    if htf_enabled:
        metrics["ev_R_htf"] = float(sig_metrics.get("ev_R", np.nan))
        metrics["n_pass_htf"] = int(len(sig_eval_df))
        base_ev = metrics.get("ev_R_baseline")
        if base_ev is not None and np.isfinite(base_ev) and abs(base_ev) > 1e-12:
            metrics["uplift_pct"] = float((metrics["ev_R_htf"] - base_ev) / abs(base_ev) * 100.0)
        else:
            metrics["uplift_pct"] = float("nan")

    # Verdict heuristic (can be refined): require uplift >= 1.7x and EV>0
    verdict = "Needs Fix"
    if np.isfinite(metrics["ev_R"]) and metrics["ev_R"] > 0:
        up_ok = (metrics.get("uplift_long_vs_up") or 0) >= 1.7
        dn_ok = (metrics.get("uplift_short_vs_down") or 0) >= 1.7
        if up_ok or dn_ok:
            verdict = "OK"
    metrics["verdict"] = verdict

    # Output directory
    if out_dir is None:
        out_dir = f"reports/patterns_{datetime.now().strftime('%Y%m%d')}"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save filtered signals
    sig_path = out_path / "signals.parquet"
    sig_eval_df.to_parquet(sig_path, index=False)

    # Additional analytics: Wilson CI and EV curve by quality thresholds
    import math

    def _wilson_ci(k: int, n: int, z: float = 1.96):
        if n <= 0:
            return (0.0, 0.0)
        phat = k / n
        denom = 1.0 + (z * z) / n
        center = (phat + (z * z) / (2 * n)) / denom
        half = z * math.sqrt((phat * (1 - phat)) / n + (z * z) / (4 * n * n)) / denom
        lo = max(0.0, center - half)
        hi = min(1.0, center + half)
        return (float(lo), float(hi))

    # Wilson CI for overall hit_rate
    k_hits = int(sig_eval_df["hit"].sum()) if len(sig_eval_df) else 0
    n_hits = int(len(sig_eval_df))
    ci_lo, ci_hi = _wilson_ci(k_hits, n_hits)
    metrics["hit_rate_ci"] = {"lo": ci_lo, "hi": ci_hi, "n": n_hits}

    # EV curve: quality thresholds
    ev_curve_path = out_path / "ev_curve.csv"
    rows_curve = []
    if {"quality", "r_net"}.issubset(set(sig_eval_df.columns)) and len(sig_eval_df) > 0:
        qs = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]
        for q in qs:
            thr = float(sig_eval_df["quality"].quantile(q)) if len(sig_eval_df) else float("nan")
            sub = sig_eval_df[sig_eval_df["quality"] >= thr]
            n = int(len(sub))
            ev = float(sub["r_net"].mean()) if n > 0 else float("nan")
            hr = float(sub["hit"].mean()) if n > 0 and "hit" in sub.columns else float("nan")
            rows_curve.append({"q": q, "quality_threshold": thr, "n": n, "EV_net": ev, "hit_rate": hr})
        pd.DataFrame(rows_curve).to_csv(ev_curve_path, index=False, encoding="utf-8")

        # Best candidate under minimum n constraint
        # evcurve_min_n provided via closure from main -> run; use default if missing
    else:
        # Write empty header for reproducibility
        pd.DataFrame(columns=["q", "quality_threshold", "n", "EV_net", "hit_rate"]).to_csv(
            ev_curve_path, index=False, encoding="utf-8"
        )

    # Choose best by EV among candidates with sufficient n
    best_obj = None
    if rows_curve:
        cand = [r for r in rows_curve if r.get("n", 0) >= int(evcurve_min_n)]
        if cand:
            # Filter out NaN EVs
            cand2 = [r for r in cand if (r.get("EV_net") is not None and not pd.isna(r.get("EV_net")))]
            if cand2:
                best_obj = max(cand2, key=lambda r: r["EV_net"])  # type: ignore
    metrics["ev_curve_best"] = (
        None
        if best_obj is None
        else {
            "q": float(best_obj["q"]),
            "quality_threshold": float(best_obj["quality_threshold"]),
            "n": int(best_obj["n"]),
            "EV_net": float(best_obj["EV_net"]),
            "hit_rate": (None if best_obj.get("hit_rate") is None or pd.isna(best_obj.get("hit_rate")) else float(best_obj["hit_rate"]))
        }
    )

    # Write metrics JSON at the end
    metrics_path = out_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Optional: save a few sample PNGs around signal points if matplotlib exists
    try:
        import matplotlib.pyplot as plt  # type: ignore

        if len(sig_df) > 0:
            samp_dir = out_path / "samples"
            samp_dir.mkdir(exist_ok=True)
            take = min(3, len(sig_df))
            for j in range(take):
                row = sig_df.iloc[j]
                i = int(row["index"]) if "index" in row else int(np.where(df.index == row["time"])[0][0])
                w = 120
                a = max(0, i - w)
                b = min(len(df) - 1, i + w)
                xs = df.index[a:b]
                ys = df["close"].iloc[a:b]
                plt.figure(figsize=(10, 4))
                plt.plot(xs, ys, label="close", color="#1f77b4")
                plt.axvline(df.index[i], color="#444", linestyle="--", alpha=0.6)
                plt.title(f"Flag/Pennant sample (idx={i}, side={int(row['side'])}, H={H})")
                # Plot delta bands at entry
                entry = float(row["entry"]) if "entry" in row else float(df["close"].iloc[i])
                delta = float(row["delta"]) if "delta" in row else float(delta_mult * df["atr"].iloc[i])
                plt.axhline(entry + delta, color="#2ca02c", linestyle=":", alpha=0.7)
                plt.axhline(entry - delta, color="#d62728", linestyle=":", alpha=0.7)
                plt.legend(loc="best")
                plt.tight_layout()
                out_png = samp_dir / f"sample_{j+1}.png"
                plt.savefig(out_png)
                plt.close()
    except Exception:
        # plotting is best-effort; ignore failures
        pass

    return metrics_path, metrics


def main():
    p = argparse.ArgumentParser(description="Evaluate Flag/Pennant pattern signals")
    p.add_argument("--data", default="data/USDJPY_15m.csv", help="Input OHLC csv")
    p.add_argument("--out", default=None, help="Output directory (default reports/patterns_YYYYMMDD)")
    p.add_argument("--H", type=int, default=8, help="Forward horizon (bars)")
    p.add_argument("--delta-mult", type=float, default=0.5, help="δ = delta_mult * ATR")
    p.add_argument("--spread-pips", type=float, default=0.5, help="Assumed spread in pips")
    p.add_argument("--pip-size", type=float, default=0.01, help="Pip size (USDJPY=0.01)")
    p.add_argument("--evcurve-min-n", type=int, default=100, help="EV曲線で閾値候補として採用する最小サンプル数")
    # Filters
    p.add_argument("--max-spread-pips", type=float, default=None, help="Exclude signals with spread above this value")
    p.add_argument("--events-csv", default=None, help="CSV with 'time' column (e.g., data/events.csv)")
    p.add_argument("--news-minutes-before", type=int, default=30)
    p.add_argument("--news-minutes-after", type=int, default=30)
    p.add_argument("--sessions-allow", default=None, help="Comma separated: e.g. 'London,NewYork'")
    p.add_argument("--ma-window", type=int, default=None, help="Trend filter: take longs above MA and shorts below MA")
    p.add_argument("--atr-pctl-min", type=float, default=None, help="ATR percentile lower bound (0-100)")
    p.add_argument("--atr-pctl-max", type=float, default=None, help="ATR percentile upper bound (0-100)")
    # HTF
    p.add_argument("--htf-enabled", action="store_true", help="Enable higher-timeframe trend alignment filter")
    p.add_argument("--htf-tf", default="1H", help="HTF timeframe (e.g., 1H)")
    p.add_argument("--htf-ma-len", type=int, default=50, help="HTF MA length")
    p.add_argument("--htf-slope-window", type=int, default=3, help="HTF MA slope diff window")
    # Configs
    p.add_argument("--patterns-yml", default="config/patterns.yml")
    p.add_argument("--patterns-session-yml", default="config/patterns_session.yml")
    args = p.parse_args()

    metrics_path, metrics = run(
        data_path=args.data,
        out_dir=args.out,
        H=args.H,
        delta_mult=args.__dict__["delta_mult"],
        spread_pips=args.__dict__["spread_pips"],
        pip_size=args.__dict__["pip_size"],
        max_spread_pips=args.__dict__["max_spread_pips"],
        events_csv=args.__dict__["events_csv"],
        news_minutes_before=args.__dict__["news_minutes_before"],
        news_minutes_after=args.__dict__["news_minutes_after"],
        sessions_allow=args.__dict__["sessions_allow"],
        ma_window=args.__dict__["ma_window"],
        atr_pctl_min=args.__dict__["atr_pctl_min"],
        atr_pctl_max=args.__dict__["atr_pctl_max"],
    htf_enabled=bool(args.__dict__["htf_enabled"]),
    htf_timeframe=str(args.__dict__["htf_tf"]),
    htf_ma_len=int(args.__dict__["htf_ma_len"]),
    htf_slope_window=int(args.__dict__["htf_slope_window"]),
    patterns_yml=args.__dict__["patterns_yml"],
    patterns_session_yml=args.__dict__["patterns_session_yml"],
        evcurve_min_n=int(args.__dict__["evcurve_min_n"]),
    )
    print(str(metrics_path))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
