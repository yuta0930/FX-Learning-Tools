from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.monitoring.observability import export_last_week, export_this_week, load_logs


def main() -> None:
    ap = argparse.ArgumentParser(description="Export weekly observability report (this/last week).")
    ap.add_argument("--logs", nargs="+", default=["logs/executions.parquet"], help="Log paths (parquet/jsonl/csv)")
    ap.add_argument("--which", choices=["this", "last"], default="this")
    ap.add_argument("--out-root", default="reports")
    ap.add_argument("--reasons-map", default=None, help="Path to reasons_map.yml for normalization")
    args = ap.parse_args()

    paths = [Path(p) for p in args.logs]
    df = load_logs(paths)

    if args.reasons_map:
        reasons_map_path = Path(args.reasons_map)
    else:
        reasons_map_path = None

    if args.which == "this":
        out = export_this_week(df, out_root=Path(args.out_root), sources=paths, reasons_map_path=reasons_map_path)
    else:
        out = export_last_week(df, out_root=Path(args.out_root), sources=paths, reasons_map_path=reasons_map_path)

    print(f"Exported to {out.resolve()}")


if __name__ == "__main__":
    main()
