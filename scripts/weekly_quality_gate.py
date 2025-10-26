"""
Weekly quality gate orchestrator

Steps:
1) Find latest patterns grid summary (reports/patterns_grid_YYYYMMDD/summary.csv)
2) Emit thresholds YAML into config/patterns_quality_thresholds.yml
3) Run 1-day health check with reasonable defaults

This script is safe to run repeatedly. It will fallback gracefully if inputs are missing.

Usage (Windows cmd):
  python scripts/weekly_quality_gate.py --tokyo-thr 0.10 --sessions "London,NewYork"

Optional args:
  --summary PATH              Specify summary.csv instead of auto-detecting the latest
  --logs PATH                 Executions log parquet path (default: logs/executions.parquet)
  --days N                    Health check days (default: 1)
  --max-block-rate FLOAT      Default 0.8
  --max-unknown-ratio FLOAT   Default 0.3
  --tokyo-thr FLOAT           Default 0.10
  --sessions CSV              Default "London,NewYork"
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from datetime import datetime
import json

try:
    import httpx  # type: ignore
except Exception:  # optional dep; skip if unavailable
    httpx = None  # type: ignore


def find_latest_summary() -> str | None:
    pattern = os.path.join("reports", "patterns_grid_*", "summary.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    # Extract date from directory name if possible and sort desc
    def key(p: str) -> tuple:
        m = re.search(r"patterns_grid_(\d{8})", p)
        if m:
            try:
                dt = datetime.strptime(m.group(1), "%Y%m%d")
            except ValueError:
                dt = datetime.min
        else:
            dt = datetime.min
        return (dt, p)

    candidates.sort(key=key, reverse=True)
    return candidates[0]


def run_cmd(cmd: list[str]) -> int:
    print(f"[weekly] Running: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=None)
    ap.add_argument("--logs", default=os.path.join("logs", "executions.parquet"))
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--max-block-rate", type=float, default=0.8, dest="max_block_rate")
    ap.add_argument("--max-unknown-ratio", type=float, default=0.3, dest="max_unknown_ratio")
    ap.add_argument("--tokyo-thr", type=float, default=0.10, dest="tokyo_thr")
    ap.add_argument("--sessions", default="London,NewYork")
    ap.add_argument("--webhook-url", default=os.getenv("WEEKLY_WEBHOOK_URL"), help="Optional: Slack/Teams webhook URL for notification")
    args = ap.parse_args(argv)

    summary = args.summary or find_latest_summary()
    if not summary or not os.path.exists(summary):
        print("[weekly] summary.csv not found. Run 'Patterns: grid' first.", file=sys.stderr)
        return 2

    # 1) Emit thresholds
    emit_cmd = [
        sys.executable,
        os.path.join("scripts", "emit_quality_thresholds.py"),
        "--summary",
        summary,
        "--sessions",
        args.sessions,
        "--tokyo-thr",
        f"{args.tokyo_thr:.2f}",
        "--out",
        os.path.join("config", "patterns_quality_thresholds.yml"),
        "--force",
    ]
    rc1 = run_cmd(emit_cmd)
    if rc1 != 0:
        print("[weekly] emit_quality_thresholds failed", file=sys.stderr)
        if args.webhook_url and httpx:
            try:
                httpx.post(args.webhook_url, json={"text": f"[weekly] emit_quality_thresholds failed for {summary}"}, timeout=10)
            except Exception:
                pass
        return rc1

    # 2) Health check
    health_cmd = [
        sys.executable,
        os.path.join("scripts", "health_check.py"),
        "--logs",
        args.logs,
        "--days",
        str(args.days),
        "--max-block-rate",
        str(args.max_block_rate),
        "--max-unknown-ratio",
        str(args.max_unknown_ratio),
        "--reasons-map",
        os.path.join("config", "reasons_map.yml"),
    ]
    rc2 = run_cmd(health_cmd)
    if rc2 != 0:
        print("[weekly] health_check failed", file=sys.stderr)
        if args.webhook_url and httpx:
            try:
                httpx.post(args.webhook_url, json={"text": f"[weekly] health_check failed (logs={args.logs}, days={args.days})"}, timeout=10)
            except Exception:
                pass
        return rc2

    print("[weekly] completed successfully.")
    if args.webhook_url and httpx:
        try:
            httpx.post(args.webhook_url, json={"text": "[weekly] completed successfully (thresholds emitted + health OK)"}, timeout=10)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
