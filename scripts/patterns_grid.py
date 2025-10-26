import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd


def run_cmd(args_list):
    print(">>", " ".join(args_list))
    r = subprocess.run(args_list, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
    return r.returncode


def main():
    # Optional: --pattern {flag_pennant|triangle|rectangle|asia_box}
    args = list(sys.argv[1:])
    pattern = "flag_pennant"
    if "--pattern" in args:
        try:
            i = args.index("--pattern")
            pattern = str(args[i+1])
            # remove tokens
            del args[i:i+2]
        except Exception:
            pass

    data = args[0] if len(args) >= 1 else "data/USDJPY_15m.csv"
    # Separate output folder per pattern to avoid overwriting summaries
    suffix = "" if pattern == "flag_pennant" else f"_{pattern}"
    out_root = Path("reports") / f"patterns_grid_{datetime.now().strftime('%Y%m%d')}{suffix}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Search grid (min-n=60) with HTF filter and hygiene constraints
    H_list = [8, 12, 16, 20]
    delta_list = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    sessions_list = ["London", "NewYork"]
    rows = []

    for H in H_list:
        for dmul in delta_list:
            for sess in sessions_list:
                run_out = out_root / f"H{H}_dm{dmul}_{sess}"
                is_fp = (pattern == "flag_pennant")
                args = [
                    sys.executable,
                    ("scripts/eval_patterns.py" if is_fp else "scripts/eval_patterns_tri_rec_asia.py"),
                ]
                if not is_fp:
                    args += ["--pattern", pattern]
                # Common args
                args += [
                    "--data", str(data),
                    "--H", str(H),
                    "--delta-mult", str(dmul),
                    "--max-spread-pips", "0.6",
                    "--events-csv", "data/events.csv",
                    "--news-minutes-before", "90",
                    "--news-minutes-after", "90",
                    "--sessions-allow", sess,
                    "--evcurve-min-n", "60",
                    "--out", str(run_out),
                ]
                # Flag/Pennant specific knobs
                if is_fp:
                    args += [
                        "--ma-window", "50",
                        "--atr-pctl-min", "20",
                        "--atr-pctl-max", "90",
                        "--htf-enabled",
                        "--htf-tf", "1H",
                        "--htf-ma-len", "50",
                        "--htf-slope-window", "3",
                    ]
                rc = run_cmd(args)
                if rc != 0:
                    continue

                mfile = run_out / "metrics.json"
                if not mfile.exists():
                    continue
                try:
                    m = json.loads(mfile.read_text(encoding="utf-8"))
                except Exception:
                    continue
                best = m.get("ev_curve_best")
                ev_net = m.get("ev_R") or m.get("EV_net") or m.get("EV_R")
                row = {
                    "H": H,
                    "delta_mult": dmul,
                    "session": sess,
                    "EV_net": ev_net,
                    "n_signals": m.get("n_signals"),
                    "hit_rate": m.get("hit_rate"),
                    "hit_rate_ci_lo": (m.get("hit_rate_ci") or {}).get("lo"),
                    "hit_rate_ci_hi": (m.get("hit_rate_ci") or {}).get("hi"),
                    "evcurve_q": None if not best else best.get("q"),
                    "evcurve_quality_thr": None if not best else best.get("quality_threshold"),
                    "evcurve_n": None if not best else best.get("n"),
                    "evcurve_EV_net": None if not best else best.get("EV_net"),
                    "out_dir": str(run_out),
                }
                # Preserve FP-specific fields if present
                for k in ["ev_R_baseline", "ev_R_htf", "uplift_pct", "n_signals_baseline", "n_pass_htf"]:
                    if k in m:
                        row[k] = m.get(k)
                rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        if pattern == "flag_pennant" and set(["ev_R_htf", "uplift_pct"]).issubset(df.columns):
            df.sort_values(by=["ev_R_htf", "uplift_pct", "EV_net"], ascending=[False, False, False], inplace=True)
        else:
            # Generic sort for the new patterns
            sort_cols = [c for c in ["EV_net", "hit_rate", "n_signals"] if c in df.columns]
            df.sort_values(by=sort_cols, ascending=[False]*len(sort_cols), inplace=True)
    out_csv = out_root / "summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Summary: {out_csv}")
    if not df.empty:
        top = df.iloc[0].to_dict()
        print("Top candidate:", top)

    # Confirmation pass on top-3 with min-n=100
    if not df.empty:
        conf_rows = []
        for _, row in df.head(3).iterrows():
            H = int(row["H"]); dmul = float(row["delta_mult"]); sess = str(row["session"]) 
            run_out = out_root / f"confirm_H{H}_dm{dmul}_{sess}"
            is_fp = (pattern == "flag_pennant")
            args = [
                sys.executable,
                ("scripts/eval_patterns.py" if is_fp else "scripts/eval_patterns_tri_rec_asia.py"),
            ]
            if not is_fp:
                args += ["--pattern", pattern]
            args += [
                "--data", str(data),
                "--H", str(H),
                "--delta-mult", str(dmul),
                "--max-spread-pips", "0.6",
                "--events-csv", "data/events.csv",
                "--news-minutes-before", "90",
                "--news-minutes-after", "90",
                "--sessions-allow", sess,
                "--evcurve-min-n", "100",
                "--out", str(run_out),
            ]
            if is_fp:
                args += [
                    "--ma-window", "50",
                    "--atr-pctl-min", "20",
                    "--atr-pctl-max", "90",
                    "--htf-enabled",
                    "--htf-tf", "1H",
                    "--htf-ma-len", "50",
                    "--htf-slope-window", "3",
                ]
            rc = run_cmd(args)
            if rc != 0:
                continue
            mfile = run_out / "metrics.json"
            if not mfile.exists():
                continue
            try:
                m = json.loads(mfile.read_text(encoding="utf-8"))
            except Exception:
                continue
            best = m.get("ev_curve_best")
            conf_rows.append({
                "H": H,
                "delta_mult": dmul,
                "session": sess,
                "EV_net": m.get("ev_R") or m.get("EV_net") or m.get("EV_R"),
                "ev_R_baseline": m.get("ev_R_baseline"),
                "ev_R_htf": m.get("ev_R_htf"),
                "uplift_pct": m.get("uplift_pct"),
                "n_pass_htf": m.get("n_pass_htf"),
                "hit_rate": m.get("hit_rate"),
                "evcurve_q": None if not best else best.get("q"),
                "evcurve_quality_thr": None if not best else best.get("quality_threshold"),
                "evcurve_n": None if not best else best.get("n"),
                "evcurve_EV_net": None if not best else best.get("EV_net"),
                "out_dir": str(run_out),
            })

        if conf_rows:
            dfc = pd.DataFrame(conf_rows)
            if pattern == "flag_pennant" and set(["ev_R_htf", "uplift_pct"]).issubset(dfc.columns):
                dfc.sort_values(by=["ev_R_htf", "uplift_pct", "EV_net"], ascending=[False, False, False], inplace=True)
            else:
                sort_cols = [c for c in ["EV_net", "hit_rate"] if c in dfc.columns]
                dfc.sort_values(by=sort_cols, ascending=[False]*len(sort_cols), inplace=True)
            out_csv2 = out_root / "summary_confirm.csv"
            dfc.to_csv(out_csv2, index=False, encoding="utf-8")
            print(f"Confirm Summary: {out_csv2}")


if __name__ == "__main__":
    main()
