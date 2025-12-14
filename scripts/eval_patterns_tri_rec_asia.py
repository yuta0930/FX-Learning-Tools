from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from src.core.ta import add_atr_if_missing  # noqa: E402
from scripts.triangle_rectangle_asia import run_detection  # noqa: E402

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["time", "timestamp", "datetime", "date"]:
            if col in df.columns:
                out = df.copy()
                out.index = pd.to_datetime(out[col], utc=True, errors="coerce")
                return out
        out = df.copy()
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        return out
    return df


def _add_session(df: pd.DataFrame) -> pd.DataFrame:
    # Simple UTC-based session bucketing (aligned with eval_patterns.py)
    def _session_of(ts: pd.Timestamp) -> str:
        h = ts.hour
        if 0 <= h <= 8:
            return "Tokyo"
        if 7 <= h <= 16:
            return "London"
        if 12 <= h <= 21:
            return "NewYork"
        return "Other"
    out = df.copy()
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

    up_first = dn_first = denom = 0
    for i in range(len(df) - H - 1):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        delta = delta_mult * atr[i]
        up = closes[i] + delta
        dn = closes[i] - delta
        hi_seg = highs[i : i + H + 1]
        lo_seg = lows[i : i + H + 1]
        _k, which = _first_hit_idx(hi_seg, lo_seg, closes[i], up, dn, H)
        denom += 1
        if which == "up":
            up_first += 1
        elif which == "down":
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
    signals: List[Dict[str, Any]],
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
        i = int(s.get("index"))
        if i + 1 >= len(df):
            continue
        if not np.isfinite(atr[i]) or atr[i] <= 1e-12:
            continue
        delta = float(s.get("delta")) if s.get("delta") is not None else (delta_mult * atr[i])
        up = closes[i] + delta
        dn = closes[i] - delta
        hi_seg = highs[i : i + H + 1]
        lo_seg = lows[i : i + H + 1]
        k, which = _first_hit_idx(hi_seg, lo_seg, closes[i], up, dn, H)
        hit = False
        r_R = 0.0
        side = int(s.get("side", 0))
        if which is not None:
            if (side > 0 and which == "up") or (side < 0 and which == "down"):
                hit = True
                r_R = 1.0
            else:
                hit = False
                r_R = -1.0
        else:
            end = min(len(df) - 1, i + H)
            ret = (closes[end] - closes[i]) if side > 0 else (closes[i] - closes[end])
            r_R = float(ret / delta)
        cost_R = (spread_pips * pv) / max(delta, 1e-12)
        r_net = r_R - cost_R
        rows.append({
            "time": df.index[i],
            "index": i,
            "side": side,
            "entry": float(closes[i]),
            "delta": float(delta),
            "H": H,
            "hit": bool(hit),
            "r_R": float(r_R),
            "r_net": float(r_net),
            "kind": str(s.get("kind", "pattern")),
            "quality": float(s.get("quality", np.nan)),
        })
    out_df = pd.DataFrame(rows)
    metrics = {
        "n_signals": int(len(out_df)),
        "hit_rate": float(out_df["hit"].mean()) if len(out_df) else float("nan"),
        "ev_R": float(out_df["r_net"].mean()) if len(out_df) else float("nan"),
    }
    return out_df, metrics


def run(
    *,
    pattern: str,
    data_path: str,
    out_dir: Optional[str],
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
    # Configs
    patterns_yml: str = "config/patterns.yml",
    patterns_session_yml: str = "config/patterns_session.yml",
    # EV curve constraint
    evcurve_min_n: int = 100,
) -> Tuple[Path, Dict[str, Any]]:
    df = pd.read_csv(data_path)
    df = _ensure_time_index(df)
    need = {"open", "high", "low", "close"}
    df.columns = [c.lower() for c in df.columns]
    if not need.issubset(df.columns):
        raise ValueError(f"dataframe must contain {need}, got {set(df.columns)}")

    df = add_atr_if_missing(df)
    df = _add_session(df)

    # Load detector cfg and overrides
    cfg_all = _load_yaml(Path(patterns_yml)) or {}
    det_cfg = (cfg_all.get(pattern, {}) if isinstance(cfg_all, dict) else {})

    sess_override_root = _load_yaml(Path(patterns_session_yml)) or {}
    # New session override style lives under Overrides -> Session -> pattern
    sess_overrides = {}
    try:
        sess_overrides = (sess_override_root.get("Overrides", {}) or {})
    except Exception:
        sess_overrides = {}

    # Detection
    # For session-specific post-filtering, we detect raw signals first, then filter by session allow-list etc.
    raw_signals = run_detection(df, pattern=pattern, cfg=det_cfg, session_override=None)
    stage_counts: Dict[str, int] = {"detected_raw": int(len(raw_signals))}

    # Build signal frame and join market info
    sig_df = pd.DataFrame(raw_signals)
    df_info = df.copy()
    # Avoid ambiguity where 'time' is both an index level name and a column label
    try:
        df_info.index.name = None
    except Exception:
        pass
    df_info["time"] = pd.to_datetime(df_info.index, utc=True)
    use_cols = [c for c in ["time", "close", "atr", "spread_pips", "session"] if c in df_info.columns]
    if not sig_df.empty:
        sig_df = sig_df.merge(df_info[use_cols], on="time", how="left")

    # A) Spread filter
    if max_spread_pips is not None and not sig_df.empty and "spread_pips" in sig_df.columns:
        before = len(sig_df)
        sig_df = sig_df[sig_df["spread_pips"].fillna(0) <= float(max_spread_pips)].copy()
        stage_counts["after_spread"] = int(len(sig_df)); stage_counts.setdefault("before_spread", before)

    # B) News window filter
    ev_windows = _load_events_windows(Path(events_csv) if events_csv else None, news_minutes_before, news_minutes_after)
    if ev_windows and not sig_df.empty:
        tvals = pd.to_datetime(sig_df["time"]).to_numpy()
        keep = []
        for tt in tvals:
            ok = True
            for a, b in ev_windows:
                if a <= tt <= b:
                    ok = False
                    break
            keep.append(ok)
        before = len(sig_df)
        sig_df = sig_df[np.array(keep)].copy()
        stage_counts["after_news"] = int(len(sig_df)); stage_counts.setdefault("before_news", before)

    # C) Session allow-list
    if sessions_allow and not sig_df.empty:
        allow = {s.strip() for s in str(sessions_allow).split(",") if s.strip()}
        if allow:
            before = len(sig_df)
            sig_df = sig_df[sig_df["session"].isin(allow)].copy()
            stage_counts["after_session_allow"] = int(len(sig_df)); stage_counts.setdefault("before_session_allow", before)

    # D) Session-specific overrides post-detection (optional)
    if not sig_df.empty and sess_overrides:
        def _ok(row):
            sess = str(row.get("session", ""))
            ov = ((sess_overrides.get(sess, {}) or {}).get(pattern, {}) or {})
            if not ov:
                return True
            # For minimal safety, apply only recognized keys
            if pattern == "triangle":
                # enforce converge_ratio_max as post filter if provided (use quality as proxy when available)
                return True
            if pattern == "rectangle":
                return True
            if pattern == "asia_box":
                return True
            return True
        before = len(sig_df)
        sig_df = sig_df[sig_df.apply(_ok, axis=1)].copy()
        stage_counts["after_session_overrides"] = int(len(sig_df)); stage_counts.setdefault("before_session_overrides", before)

    # Evaluate
    filt_idx = set(int(i) for i in sig_df.get("index", []).tolist()) if not sig_df.empty else set()
    signals_f = [s for s in raw_signals if int(s.get("index")) in filt_idx]

    sig_eval_df, sig_metrics = _eval_signals(df, signals_f, H, delta_mult, spread_pips, pip_size)
    baseline = _compute_baseline_rates(df, H, delta_mult, pip_size=pip_size)

    # Uplift calc per direction
    if len(sig_eval_df):
        pos = sig_eval_df[sig_eval_df["side"] > 0]
        neg = sig_eval_df[sig_eval_df["side"] < 0]
        hit_pos = float(pos["hit"].mean()) if len(pos) else np.nan
        hit_neg = float(neg["hit"].mean()) if len(neg) else np.nan
    else:
        hit_pos = hit_neg = np.nan

    uplift_pos = (hit_pos / baseline["up_rate"]) if (np.isfinite(hit_pos) and baseline["up_rate"] not in (0, np.nan)) else np.nan
    uplift_neg = (hit_neg / baseline["down_rate"]) if (np.isfinite(hit_neg) and baseline["down_rate"] not in (0, np.nan)) else np.nan

    metrics: Dict[str, Any] = {
        "pattern": pattern,
        "H": H,
        "delta_mult": delta_mult,
        "spread_pips": spread_pips,
        "n_rows": int(len(df)),
        "n_signals": int(len(sig_eval_df)),
        "hit_rate": float(sig_metrics.get("hit_rate", float("nan"))),
        "ev_R": float(sig_metrics.get("ev_R", float("nan"))),
        "baseline_up": float(baseline.get("up_rate", float("nan"))),
        "baseline_down": float(baseline.get("down_rate", float("nan"))),
        "uplift_long_vs_up": float(uplift_pos) if np.isfinite(uplift_pos) else float("nan"),
        "uplift_short_vs_down": float(uplift_neg) if np.isfinite(uplift_neg) else float("nan"),
        "stage_counts": stage_counts,
    }

    # Verdict rubric
    verdict = "Needs Fix"
    if np.isfinite(metrics["ev_R"]) and metrics["ev_R"] > 0:
        up_ok = (metrics.get("uplift_long_vs_up") or 0) >= 1.7
        dn_ok = (metrics.get("uplift_short_vs_down") or 0) >= 1.7
        if up_ok or dn_ok:
            verdict = "OK"
    metrics["verdict"] = verdict

    # Output dir
    if out_dir is None:
        out_dir = f"reports/patterns_{datetime.now().strftime('%Y%m%d')}"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save signals
    sig_path = out_path / "signals.parquet"
    sig_eval_df.to_parquet(sig_path, index=False)

    # Wilson CI and EV curve
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

    k_hits = int(sig_eval_df["hit"].sum()) if len(sig_eval_df) else 0
    n_hits = int(len(sig_eval_df))
    ci_lo, ci_hi = _wilson_ci(k_hits, n_hits)
    metrics["hit_rate_ci"] = {"lo": ci_lo, "hi": ci_hi, "n": n_hits}

    ev_curve_path = out_path / "ev_curve.csv"
    rows_curve: List[Dict[str, Any]] = []
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
    else:
        pd.DataFrame(columns=["q", "quality_threshold", "n", "EV_net", "hit_rate"]).to_csv(ev_curve_path, index=False, encoding="utf-8")

    best_obj = None
    if rows_curve:
        cand = [r for r in rows_curve if r.get("n", 0) >= int(evcurve_min_n)]
        cand2 = [r for r in cand if (r.get("EV_net") is not None and not pd.isna(r.get("EV_net")))]
        if cand2:
            best_obj = max(cand2, key=lambda r: r["EV_net"])  # type: ignore
    metrics["ev_curve_best"] = None if best_obj is None else {
        "q": float(best_obj["q"]),
        "quality_threshold": float(best_obj["quality_threshold"]),
        "n": int(best_obj["n"]),
        "EV_net": float(best_obj["EV_net"]),
        "hit_rate": (None if best_obj.get("hit_rate") is None or pd.isna(best_obj.get("hit_rate")) else float(best_obj["hit_rate"]))
    }

    metrics_path = out_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Session summary (optional and safe)
    try:
        if len(sig_eval_df):
            df_info2 = df[["session"]].copy()
            # Bring index out as a column named 'time' robustly
            df_info2 = df_info2.reset_index()
            # Ensure the first column (former index) is named 'time'
            idx_col = df_info2.columns[0]
            if idx_col != "time":
                df_info2 = df_info2.rename(columns={idx_col: "time"})
            df_info2["time"] = pd.to_datetime(df_info2["time"], utc=True)
            joined = sig_eval_df.merge(df_info2[["time", "session"]], on="time", how="left")
            if "session" in joined.columns:
                g = joined.groupby("session")
                sess_rows = []
                for sess, sub in g:
                    n = int(len(sub))
                    hr = float(sub["hit"].mean()) if n else float("nan")
                    ev = float(sub["r_net"].mean()) if n else float("nan")
                    sess_rows.append({"session": sess, "n": n, "hit_rate": hr, "EV_net": ev})
                pd.DataFrame(sess_rows).to_csv(out_path / "session_summary.csv", index=False, encoding="utf-8")
        else:
            # create an empty schema for reproducibility
            pd.DataFrame(columns=["session", "n", "hit_rate", "EV_net"]).to_csv(
                out_path / "session_summary.csv", index=False, encoding="utf-8"
            )
    except Exception:
        # best-effort
        pass

    return metrics_path, metrics


def main():
    p = argparse.ArgumentParser(description="Evaluate Triangle/Rectangle/Asia Box pattern signals")
    p.add_argument("--pattern", required=True, choices=["triangle", "rectangle", "asia_box"])
    p.add_argument("--data", default="data/USDJPY_15m.csv")
    p.add_argument("--out", default=None)
    p.add_argument("--H", type=int, default=12)
    p.add_argument("--delta-mult", type=float, default=0.6)
    p.add_argument("--spread-pips", type=float, default=0.5)
    p.add_argument("--pip-size", type=float, default=0.01)
    p.add_argument("--evcurve-min-n", type=int, default=100)
    # Filters
    p.add_argument("--max-spread-pips", type=float, default=None)
    p.add_argument("--events-csv", default=None)
    p.add_argument("--news-minutes-before", type=int, default=45)
    p.add_argument("--news-minutes-after", type=int, default=45)
    p.add_argument("--sessions-allow", default=None)
    # Configs
    p.add_argument("--patterns-yml", default="config/patterns.yml")
    p.add_argument("--patterns-session-yml", default="config/patterns_session.yml")
    args = p.parse_args()

    metrics_path, metrics = run(
        pattern=str(args.pattern),
        data_path=args.data,
        out_dir=args.out,
        H=int(args.H),
        delta_mult=float(args.__dict__["delta_mult"]),
        spread_pips=float(args.__dict__["spread_pips"]),
        pip_size=float(args.__dict__["pip_size"]),
        max_spread_pips=args.__dict__["max_spread_pips"],
        events_csv=args.__dict__["events_csv"],
        news_minutes_before=int(args.__dict__["news_minutes_before"]),
        news_minutes_after=int(args.__dict__["news_minutes_after"]),
        sessions_allow=args.__dict__["sessions_allow"],
        patterns_yml=args.__dict__["patterns_yml"],
        patterns_session_yml=args.__dict__["patterns_session_yml"],
        evcurve_min_n=int(args.__dict__["evcurve_min_n"]),
    )
    print(str(metrics_path))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
