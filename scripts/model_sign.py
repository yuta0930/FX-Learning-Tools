from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute SHA-256 of model file and write into models/break_meta.json")
    ap.add_argument("--model", default=None, help="Path to model file (default: choose calibrated if exists, else break_model.joblib)")
    ap.add_argument("--meta", default=os.path.join("models", "break_meta.json"), help="Meta json path to write")
    args = ap.parse_args()

    # choose model path
    model_path = args.model
    if not model_path:
        cal = os.path.join("models", "break_model_calibrated.joblib")
        base = os.path.join("models", "break_model.joblib")
        model_path = cal if os.path.exists(cal) else base
    if not os.path.exists(model_path):
        print(f"model not found: {model_path}")
        return 2

    sig = sha256_of_file(model_path)
    now = datetime.now(timezone.utc).isoformat()

    # merge into meta json
    meta = {}
    if os.path.exists(args.meta):
        try:
            with open(args.meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    meta["model_path"] = model_path
    meta["model_sha256"] = sig
    meta["signed_at_utc"] = now

    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps({"model": model_path, "sha256": sig, "meta": args.meta}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
