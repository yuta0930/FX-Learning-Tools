"""
Update or create a price CSV using yfinance.

Usage (examples):
  env/Scripts/python.exe scripts/update_price_csv.py --symbol JPY=X --interval 15m --period 60d --out data/USDJPY_15m.csv

Notes:
  - This script overwrites the target CSV by merging the latest download with any existing file, de-duplicated by timestamp.
  - The output columns will be similar to existing files: timestamp, open, high, low, close, Adj Close, volume, Dividends, Stock Splits.
  - Timestamps are emitted in UTC with timezone info where possible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import timezone

import pandas as pd


def _download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as e:
        print("yfinance のインポートに失敗しました。requirements.txt に yfinance を追加/インストールしてください。", file=sys.stderr)
        raise

    df = yf.download(symbol, interval=interval, period=period, auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance からデータを取得できませんでした。")

    # Handle MultiIndex columns (e.g., when yfinance returns columns like ('Open','JPY=X'))
    try:
        import pandas as _pd  # local alias
        if isinstance(df.columns, _pd.MultiIndex):
            # If symbol level exists, try to select it; else flatten to level 0
            lev_names = list(df.columns.names or [])
            if len(df.columns.levels) >= 2:
                sym_levels = df.columns.get_level_values(-1)
                # Prefer exact match, else first non-empty
                if symbol in list(sym_levels):
                    try:
                        df = df.xs(symbol, axis=1, level=-1, drop_level=True)
                    except Exception:
                        df.columns = df.columns.get_level_values(0)
                else:
                    df.columns = df.columns.get_level_values(0)
            else:
                df.columns = df.columns.get_level_values(0)
    except Exception:
        pass

    df = df.reset_index()
    # Normalize column names to lower snake case expected by app
    rename_map = {
        "Datetime": "timestamp",
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "Adj Close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # Ensure timestamp tz-aware in UTC
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(timezone.utc)

    # Optional columns to align with existing CSVs
    if "Adj Close" not in df.columns and "close" in df.columns:
        df["Adj Close"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0
    # Add placeholders to match existing schema if present
    if "Dividends" not in df.columns:
        df["Dividends"] = 0.0
    if "Stock Splits" not in df.columns:
        df["Stock Splits"] = 0.0

    # Column order similar to existing dataset
    cols = [
        c
        for c in [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "Adj Close",
            "volume",
            "Dividends",
            "Stock Splits",
        ]
        if c in df.columns
    ]
    others = [c for c in df.columns if c not in cols]
    return df[cols + others]


def _merge_existing(out_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    if not out_path.exists():
        return new_df
    try:
        # Read existing CSV (supports timestamp with/without tz)
        old = pd.read_csv(out_path)
        if "timestamp" in old.columns:
            old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True, errors="coerce")
    except Exception:
        # Fallback: if corrupted, just return new data
        return new_df

    # Union by columns; align and concat then drop duplicates by timestamp
    all_cols = sorted(set(old.columns).union(set(new_df.columns)))
    old2 = old.reindex(columns=all_cols)
    new2 = new_df.reindex(columns=all_cols)
    merged = pd.concat([old2, new2], ignore_index=True)
    if "timestamp" in merged.columns:
        merged = merged.sort_values("timestamp")
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
    else:
        merged = merged.drop_duplicates(keep="last")
    return merged


def main():
    ap = argparse.ArgumentParser(description="Update/create price CSV via yfinance")
    ap.add_argument("--symbol", default="JPY=X", help="yfinance symbol (default: JPY=X for USDJPY)")
    ap.add_argument("--interval", default="15m", help="bar interval, e.g., 1m, 5m, 15m, 1h, 1d")
    ap.add_argument("--period", default="60d", help="lookback period for download, e.g., 7d, 30d, 60d, 2y")
    ap.add_argument("--out", default="data/USDJPY_15m.csv", help="output CSV path")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = _download(args.symbol, args.interval, args.period)
    merged = _merge_existing(out_path, new_df)

    # Emit with RFC3339-like timestamp string
    if "timestamp" in merged.columns:
        # Keep tz-aware in UTC; pandas will render as ISO8601 with Z when using to_csv if dtype is datetime64[ns, UTC]
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)

    merged.to_csv(out_path, index=False)
    if "timestamp" in merged.columns and len(merged) > 0:
        ts = merged["timestamp"].iloc[-1]
        print(f"Wrote {len(merged)} rows to {out_path} (last bar UTC: {ts})")
    else:
        print(f"Wrote {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
