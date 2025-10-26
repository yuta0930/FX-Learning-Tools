from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from config.loader import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate config/config.yml (or FX_CFG) against Pydantic schema")
    ap.add_argument("--path", default=os.environ.get("FX_CFG") or "config/config.yml")
    ap.add_argument("--strict", action="store_true", help="Fail with non-zero exit if validation fails")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.path, validate=True, strict=args.strict)
        print(json.dumps({"ok": True, "path": args.path, "sample": list(cfg.to_dict().keys())[:8]}, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "path": args.path, "error": str(e)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
