from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def session_of(ts: pd.Timestamp) -> str:
    h = ts.hour
    if 9 <= h < 15:
        return "Tokyo"
    if 16 <= h < 24:
        return "London"
    if h >= 22 or h < 5:
        return "NY"
    return "Other"


def run(signals_path: str, out_csv: str | None = None) -> pd.DataFrame:
    sigs = pd.read_parquet(signals_path)
    if "time" in sigs.columns:
        sigs["time"] = pd.to_datetime(sigs["time"])  # ensure ts
        sess = sigs["time"].dt.tz_localize(None).apply(session_of)
    else:
        # Fallback if only index exists
        sigs = sigs.reset_index()
        sigs["time"] = pd.to_datetime(sigs["time"])  # type: ignore
        sess = sigs["time"].dt.tz_localize(None).apply(session_of)

    sigs["session"] = sess
    grp = sigs.groupby("session").agg(
        n=("hit", "size"),
        hit_rate=("hit", "mean"),
        ev_R=("r_net", "mean"),
        quality_p50=("quality", "median"),
    ).reset_index()
    grp = grp.sort_values("n", ascending=False)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        grp.to_csv(out_csv, index=False)
    return grp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", default="reports/patterns_00000000/signals.parquet")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    df = run(args.signals, args.out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
