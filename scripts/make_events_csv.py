from __future__ import annotations

from pathlib import Path
import argparse
import yaml

import _bootstrap  # noqa: F401
from src.core.events import load_events_yaml, events_to_df, write_events_csv
from src.core.events_schema import validate_events_payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate data/events.csv from config/events.yml")
    ap.add_argument("--in", dest="in_path", default="config/events.yml", help="YAML input path")
    ap.add_argument("--out", dest="out_path", default="data/events.csv", help="CSV output path")
    ap.add_argument("--strict", action="store_true", help="Validate events.yml strictly (fail on invalid)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    # Strict validation (optional)
    if args.strict:
        data = yaml.safe_load(in_path.read_text(encoding="utf-8")) or {}
        # Will raise on invalid and cause non-zero exit
        validate_events_payload(data)

    items = load_events_yaml(in_path)
    df = events_to_df(items)
    write_events_csv(df, out_path)
    print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
