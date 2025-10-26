import sys
import json
import csv
import shutil
from pathlib import Path
import argparse

try:
    import yaml  # type: ignore
except Exception as e:
    yaml = None


def load_evcurve_best_from_metrics(p: Path) -> float:
    js = json.loads(p.read_text(encoding="utf-8"))
    best = js.get("ev_curve_best")
    if not best or best.get("quality_threshold") is None:
        raise ValueError("metrics.json に ev_curve_best.quality_threshold が見つかりません。")
    return float(best["quality_threshold"])


def load_best_from_summary(p: Path) -> float | None:
    """summary.csv の先頭行から evcurve_quality_thr を読み取る。

    互換性のため、列が存在しない場合は None を返す（呼び出し側でフォールバック処理）。
    """
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        row = next(r, None)
    if not row:
        return None
    val = row.get("evcurve_quality_thr")
    if val is None or str(val).strip() == "" or str(val).lower() == "none":
        return None
    try:
        return float(val)  # type: ignore
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit session-wise pattern quality thresholds YAML from metrics or summary")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--metrics", type=str, help="reports/.../metrics.json")
    src.add_argument("--summary", type=str, help="reports/.../summary.csv（先頭行を採用）")
    ap.add_argument("--pattern", type=str, default=None, help="しきい値を書き込むパターン名（指定時はネスト形式で出力）")
    ap.add_argument("--sessions", type=str, default="London,NewYork", help="適用セッション（カンマ区切り）")
    ap.add_argument("--tokyo-thr", type=float, default=None, help="Tokyo のしきい値（任意。指定時のみ出力）")
    ap.add_argument("--fallback-thr", type=float, default=0.0, help="summary に evcurve_quality_thr が無い場合のフォールバック値")
    ap.add_argument("--default-thr", type=float, default=None, help="default のしきい値（任意）")
    ap.add_argument("--out", type=str, default="config/patterns_quality_thresholds.yml")
    ap.add_argument("--force", action="store_true", help="既存YAMLを上書き（バックアップ作成）")
    args = ap.parse_args()

    if yaml is None:
        print("PyYAML が見つかりません。pip install pyyaml を実行してください。", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.metrics:
        qthr = load_evcurve_best_from_metrics(Path(args.metrics))
    else:
        qthr_loaded = load_best_from_summary(Path(args.summary))
        if qthr_loaded is None:
            # フォールバック値を使用（デフォルト 0.0）
            qthr = float(args.fallback_thr)
            print(
                f"[emit_quality_thresholds] summary に evcurve_quality_thr が無いためフォールバック {qthr} を使用: {args.summary}",
                file=sys.stderr,
            )
        else:
            qthr = float(qthr_loaded)

    sessions = [s.strip() for s in str(args.sessions).split(",") if s.strip()]
    flat_map: dict[str, float] = {}
    for s in sessions:
        flat_map[s] = float(qthr)
    if args.tokyo_thr is not None:
        flat_map["Tokyo"] = float(args.tokyo_thr)
    if args.default_thr is not None:
        flat_map["default"] = float(args.default_thr)

    # Build output data (flat or nested by pattern)
    if args.pattern:
        data: dict[str, dict[str, float]] = {str(args.pattern): flat_map}
    else:
        data = flat_map  # type: ignore

    if out.exists() and not args.force:
        # 既存を軽くマージ（既存キーは上書きし、それ以外は残す）
        try:
            prev = yaml.safe_load(out.read_text(encoding="utf-8")) or {}
        except Exception:
            prev = {}
        if isinstance(prev, dict):
            # Nested-aware merge
            if args.pattern and isinstance(data, dict) and isinstance(prev.get(args.pattern), dict):
                # merge under specific pattern
                prev_pat = prev.get(args.pattern) or {}
                prev_pat.update(data[args.pattern])
                prev[args.pattern] = prev_pat
                data = prev  # type: ignore
            elif args.pattern and isinstance(data, dict):
                prev.update(data)
                data = prev  # type: ignore
            else:
                prev.update(data)  # type: ignore
                data = prev  # type: ignore
    elif out.exists() and args.force:
        try:
            shutil.copy2(out, out.with_suffix(out.suffix + ".bak"))
        except Exception:
            pass

    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
