import os
import json
import csv
from datetime import datetime

import joblib


def load_meta(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_model_history(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else None
    except Exception:
        return None


def main():
    model_path = os.path.join("models", "break_model.joblib")
    meta_path = os.path.join("models", "break_meta.json")
    hist_path = os.path.join("reports", "model_history.csv")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print("[check] model or meta not found. model=", os.path.exists(model_path), " meta=", os.path.exists(meta_path))
        return

    # 1) use_cols に lvl_* が含まれるか
    pkg = joblib.load(model_path)
    use_cols = pkg.get("use_cols") or pkg.get("Xcols") or []
    lvl_cols = [c for c in use_cols if isinstance(c, str) and c.startswith("lvl_")]
    print("[use_cols] n=", len(use_cols))
    print("[use_cols] lvl_cols_n=", len(lvl_cols))
    if lvl_cols:
        print("[use_cols] lvl_cols=", lvl_cols[:12])
    else:
        print("[use_cols] no lvl_* columns found")

    # 2) メタのOOF指標
    meta = load_meta(meta_path)
    oof = meta.get("OOF", {}) or {}
    ap = oof.get("AP_macro")
    br = oof.get("Brier_macro")
    print("[meta] OOF AP_macro=", ap, " Brier_macro=", br)

    # 3) 履歴CSVとの整合（存在すれば比較）
    last = read_model_history(hist_path)
    if last is not None:
        try:
            ap_hist = None if last.get("AP_macro") in (None, "nan", "") else float(last.get("AP_macro"))
            br_hist = None if last.get("Brier_macro") in (None, "nan", "") else float(last.get("Brier_macro"))
        except Exception:
            ap_hist = br_hist = None
        ok_ap = (ap is None and ap_hist is None) or (ap is not None and ap_hist is not None and abs(float(ap) - float(ap_hist)) < 1e-6)
        ok_br = (br is None and br_hist is None) or (br is not None and br_hist is not None and abs(float(br) - float(br_hist)) < 1e-6)
        print(f"[history] last status={last.get('status')} AP_macro={ap_hist} Brier_macro={br_hist}")
        print(f"[consistency] AP_match={ok_ap} Brier_match={ok_br}")
    else:
        print("[history] not found or empty; skipping consistency check with history")

    # 4) 変更時刻の表示（デバッグ用）
    mt = datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
    mm = datetime.fromtimestamp(os.path.getmtime(meta_path)).isoformat()
    print(f"[mtime] model={mt} meta={mm}")


if __name__ == "__main__":
    main()
