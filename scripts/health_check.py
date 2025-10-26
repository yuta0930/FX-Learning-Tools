from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

from src.monitoring.health import assess_health, HealthThresholds
from src.monitoring.failure_log import log_failure
import os
try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Simple health check (block rate / unknown reasons / optional drift). Non-zero exit on failure."
    )
    ap.add_argument("--logs", nargs="+", default=["logs/executions.parquet"], help="Log paths")
    ap.add_argument("--days", type=int, default=1, help="Lookback days")
    ap.add_argument("--max-block-rate", type=float, default=0.8)
    ap.add_argument("--max-unknown-ratio", type=float, default=0.3)
    ap.add_argument("--reasons-map", default=None, help="Path to reasons_map.yml (optional)")
    # Drift options (optional)
    ap.add_argument("--drift-source", default=None, help="Path to series file (csv/parquet) for drift check (optional)")
    ap.add_argument("--drift-column", default="close", help="Column name for drift series")
    ap.add_argument("--drift-ref-n", type=int, default=2000, help="Reference window length")
    ap.add_argument("--drift-cur-n", type=int, default=500, help="Current window length")
    ap.add_argument("--drift-bins", type=int, default=20, help="Histogram bins for drift")
    ap.add_argument("--max-drift-psi", type=float, default=None, help="Max allowed PSI (optional: unset to skip)")
    ap.add_argument("--max-drift-js", type=float, default=None, help="Max allowed JS divergence (optional: unset to skip)")
    ap.add_argument("--logs-dir", default="logs", help="Directory to append gate failure records (jsonl). Optional.")
    ap.add_argument("--webhook-url", default=os.getenv("HEALTH_WEBHOOK_URL"), help="Optional: Slack/Teams webhook URL for immediate notification")
    args = ap.parse_args()

    th = HealthThresholds(
        lookback_days=args.days,
        max_block_rate=args.max_block_rate,
        max_unknown_ratio=args.max_unknown_ratio,
        max_drift_psi=args.max_drift_psi,
        max_drift_js=args.max_drift_js,
    )
    paths = [Path(p) for p in args.logs]
    res = assess_health(
        paths,
        th,
        Path(args.reasons_map) if args.reasons_map else None,
        drift_source=(Path(args.drift_source) if args.drift_source else None),
        drift_column=str(args.drift_column),
        drift_ref_n=int(args.drift_ref_n),
        drift_cur_n=int(args.drift_cur_n),
        drift_bins=int(args.drift_bins),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if not res.get("ok"):
        try:
            log_failure(
                logs_dir=str(args.logs_dir),
                reason="health_gate_failed",
                details={
                    "max_block_rate": args.max_block_rate,
                    "max_unknown_ratio": args.max_unknown_ratio,
                    "result": res,
                },
                hint="Inspect Observability page and recent logs; adjust thresholds if appropriate.",
            )
        except Exception:
            pass
        # Optional immediate notification on failure
        if args.webhook_url and httpx:
            try:
                httpx.post(
                    args.webhook_url,
                    json={"text": f"[health] FAILED: block_rate<= {args.max_block_rate}, unknown<= {args.max_unknown_ratio}. See logs."},
                    timeout=10,
                )
            except Exception:
                pass
    else:
        if args.webhook_url and httpx:
            try:
                httpx.post(args.webhook_url, json={"text": "[health] OK"}, timeout=10)
            except Exception:
                pass
    sys.exit(0 if res.get("ok") else 2)


if __name__ == "__main__":
    main()
