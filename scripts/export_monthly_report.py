from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from src.monitoring.observability import export_monthly_report, load_logs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export monthly observability report to reports/monthly_YYYYMM/"
    )
    ap.add_argument(
        "--logs",
        nargs="+",
        default=["logs/executions.parquet"],
        help="Log paths (parquet/jsonl/csv)",
    )
    ap.add_argument(
        "--as-of",
        default=None,
        help="As-of date (YYYY-MM-DD), default: today",
    )
    args = ap.parse_args()

    paths = [Path(p) for p in args.logs]
    df = load_logs(paths)
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else None
    out_dir = export_monthly_report(df, as_of=as_of)
    print(f"Exported to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
