"""Grid-like optimization for Flag/Pennant detector on 15m intraday data.
Usage:
  python scripts/optimize_flag_pennant_15m.py --data data/USDJPY_15m.csv --out reports/opt_flag_pennant_15m

Focus:
  London & NewYork sessions (intraday continuation setups)
Metrics:
  hit_rate, ev_R (net expectancy in R units), n_signals
Selection:
  Maximizes ev_R subject to minimum n_signals and hit_rate stability.

Notes:
  This is a lightweight in-repo optimizer to iterate detector hyperparameters
  without shelling out to the heavier eval_patterns grid (faster inner loop).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
import itertools
import math
import json
import traceback

import numpy as np
import pandas as pd

# Make repository paths importable when running as a script
import sys, os
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
_SRC = os.path.join(_ROOT, "src")
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# Local imports (after path setup)
from src.patterns.flag_pennant import detect_flag_pennant_simplified, FlagPennantSignal
from src.core.ta import add_atr_if_missing  # assuming this exists per eval_patterns.py


def ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name is None or not np.issubdtype(df.index.dtype, np.datetime64):
        for c in ["time", "timestamp", "datetime", "date"]:
            if c in df.columns:
                d2 = df.copy(); d2.index = pd.to_datetime(d2[c]); return d2
        d2 = df.copy(); d2.index = pd.to_datetime(d2.index); return d2
    return df


def add_session(df: pd.DataFrame) -> pd.DataFrame:
    def _sess(ts: pd.Timestamp) -> str:
        h = ts.hour
        if 0 <= h <= 8: return "Tokyo"
        if 7 <= h <= 16: return "London"
        if 12 <= h <= 21: return "NewYork"
        return "Other"
    out = df.copy(); out["session"] = pd.to_datetime(out.index).map(_sess); return out


def first_barrier_hit(highs: np.ndarray, lows: np.ndarray, entry: float, up: float, dn: float, H: int):
    for k in range(1, H+1):
        if k >= len(highs): break
        if highs[k] >= up: return k, "up"
        if lows[k] <= dn: return k, "down"
    return None, None


def eval_signals(df: pd.DataFrame, signals: list[FlagPennantSignal], H: int, delta_mult: float, spread_pips: float, pip_size: float):
    highs = pd.to_numeric(df["high"], errors="coerce").astype(float).values
    lows  = pd.to_numeric(df["low"], errors="coerce").astype(float).values
    closes= pd.to_numeric(df["close"], errors="coerce").astype(float).values
    atr   = pd.to_numeric(df["atr"], errors="coerce").astype(float).values
    pv = pip_size
    rows = []
    for s in signals:
        i = int(s.index)
        if i+1 >= len(df):
            continue
        if not np.isfinite(atr[i]) or atr[i] <= 1e-12:
            continue
        delta = delta_mult * atr[i]
        up = closes[i] + delta
        dn = closes[i] - delta
        hi_seg = highs[i:i+H+1]
        lo_seg = lows[i:i+H+1]
        k, which = first_barrier_hit(hi_seg, lo_seg, closes[i], up, dn, H)
        hit = False; r_R = 0.0
        if which is not None:
            if (s.side > 0 and which == "up") or (s.side < 0 and which == "down"):
                hit = True; r_R = 1.0
            else:
                hit = False; r_R = -1.0
        else:
            end = min(len(df)-1, i+H)
            ret = (closes[end] - closes[i]) if s.side > 0 else (closes[i] - closes[end])
            r_R = float(ret / delta)
        cost_R = (spread_pips * pv) / delta
        r_net = r_R - cost_R
        rows.append({"time": df.index[i], "index": i, "side": s.side, "H": H, "delta": delta, "hit": hit, "r_R": r_R, "r_net": r_net, "quality": s.quality})
    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"n_signals": 0, "hit_rate": float("nan"), "ev_R": float("nan")}
    return out, {"n_signals": len(out), "hit_rate": float(out["hit"].mean()), "ev_R": float(out["r_net"].mean())}


@dataclass
class ParamResult:
    params: dict
    n_signals: int
    hit_rate: float
    ev_R: float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/USDJPY_15m.csv")
    ap.add_argument("--out", default="reports/opt_flag_pennant_15m")
    ap.add_argument("--H", type=int, default=8)
    # store hyphenated CLI into attribute delta_mult explicitly
    ap.add_argument("--delta-mult", dest="delta_mult", type=float, default=0.5)
    ap.add_argument("--spread-pips", dest="spread_pips", type=float, default=0.5)
    ap.add_argument("--pip-size", dest="pip_size", type=float, default=0.01)
    ap.add_argument("--min-signals", type=int, default=40)
    ap.add_argument("--min-hit", type=float, default=0.45)
    ap.add_argument("--sessions", default="London,NewYork")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    # early log to confirm execution
    (out_dir / "_run.log").write_text("start\n", encoding="utf-8")
    with (out_dir / "_run.log").open("a", encoding="utf-8") as fh:
        fh.write("args=" + str(vars(args)) + "\n")
    df = pd.read_csv(args.data)
    df = ensure_time_index(df)
    df = add_atr_if_missing(df)
    df = add_session(df)

    sessions = {s.strip() for s in args.sessions.split(",") if s.strip()}
    df_sess = df[df["session"].isin(sessions)].copy()

    # Parameter space (manageable size ~72 combos)
    space = dict(
        lookback=[1000],
        n_push=[24, 30, 36],
        min_flag_bars=[8],
        max_flag_bars=[32, 40],
        sigma_k=[1.2],
        pole_min_atr=[2.5, 3.0, 3.5],
        flag_slope_max_atr=[0.08, 0.10],
        contraction_percentile=[0.10, 0.15],
    )

    keys = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))
    results: list[ParamResult] = []

    # Debug dump of argparse namespace keys
    try:
        with (out_dir / "_run.log").open("a", encoding="utf-8") as fh:
            fh.write("namespace_keys=" + ",".join(sorted(vars(args).keys())) + "\n")
    except Exception:
        pass

    total = len(combos)
    processed = 0
    for combo in combos:
        p = {k: v for k, v in zip(keys, combo)}
        try:
            sigs = detect_flag_pennant_simplified(df_sess, **p)
            eval_df, metrics = eval_signals(
                df_sess,
                sigs,
                args.H,
                args.delta_mult,
                args.spread_pips,
                args.pip_size,
            )
            results.append(
                ParamResult(
                    params=p,
                    n_signals=metrics["n_signals"],
                    hit_rate=metrics["hit_rate"],
                    ev_R=metrics["ev_R"],
                )
            )
        except Exception as e:
            with (out_dir / "_run.log").open("a", encoding="utf-8") as fh:
                fh.write(f"combo {p} error: {e.__class__.__name__}: {e}\n")
                fh.write(traceback.format_exc() + "\n")
            continue
        processed += 1
        if processed % 10 == 0 or processed == total:
            try:
                with (out_dir / "_run.log").open("a", encoding="utf-8") as fh:
                    fh.write(f"progress {processed}/{total}\n")
            except Exception:
                pass

    rows = []
    for r in results:
        row = dict(r.params)
        row.update(n_signals=r.n_signals, hit_rate=r.hit_rate, ev_R=r.ev_R)
        rows.append(row)
    res_df = pd.DataFrame(rows)
    res_path = out_dir / "grid_results.csv"; res_df.to_csv(res_path, index=False, encoding="utf-8")

    # Pick best under constraints
    cand = res_df[(res_df.n_signals >= args.min_signals) & (res_df.hit_rate >= args.min_hit)].copy()
    best = None
    if not cand.empty:
        # Tie-break: higher ev_R then higher hit_rate then more signals
        cand.sort_values(by=["ev_R", "hit_rate", "n_signals"], ascending=[False, False, False], inplace=True)
        best = cand.iloc[0].to_dict()
    best_path = out_dir / "best.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({"best": best, "constraints": {"min_signals": args.min_signals, "min_hit": args.min_hit}}, f, ensure_ascii=False, indent=2)

    print(f"Saved grid: {res_path}")
    if res_df.empty:
        print("[warn] grid_results is empty (no signals or all combos failed). Check _run.log and session filter.")
    print(f"Best: {best_path}")
    if best:
        print("Best params:\n" + json.dumps(best, ensure_ascii=False, indent=2))
    else:
        print("No candidate met constraints; inspect grid_results.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Best-effort global logging
        try:
            Path("reports/opt_flag_pennant_15m").mkdir(parents=True, exist_ok=True)
            Path("reports/opt_flag_pennant_15m/_run.log").write_text(f"fatal: {e}", encoding="utf-8")
        except Exception:
            pass
        raise
