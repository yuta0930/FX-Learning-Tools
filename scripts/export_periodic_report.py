from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from src.monitoring.observability import export_periodic_report, load_logs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export periodic observability report to reports/periodic_YYYYMMDD_YYYYMMDD/"
    )
    ap.add_argument(
        "--logs",
        nargs="+",
        default=["logs/executions.parquet"],
        help="Log paths (parquet/jsonl/csv)",
    )
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--out-root", default="reports", help="Output root directory")
    args = ap.parse_args()

    paths = [Path(p) for p in args.logs]
    df = load_logs(paths)
    start = datetime.fromisoformat(args.start).date()
    end = datetime.fromisoformat(args.end).date()

    out_dir = export_periodic_report(
        df, start=start, end=end, out_root=Path(args.out_root), sources=paths
    )
    print(f"Exported to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
