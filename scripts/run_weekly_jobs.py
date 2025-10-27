from __future__ import annotations
import os
import yaml
import pandas as pd
from app.calibration.online_calibrator import OnlineCalibrator
from app.tca.tca_runner import fit_from_logs
from app.ab.ab_manager import ABManager
from constants import SIGNALS_LOG


# ---- Weekly KPI report (lightweight) ----
def _ece(probs, labels, n_bins=10):
    import numpy as np
    import pandas as pd
    probs = np.asarray(probs); labels = np.asarray(labels)
    if len(probs) == 0 or len(labels) == 0:
        return float('nan')
    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(probs, qs)
    bins[0], bins[-1] = 0.0, 1.0
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            m = (probs >= bins[i]) & (probs < bins[i+1])
        else:
            m = (probs >= bins[i]) & (probs <= bins[i+1])
        if not m.any():
            continue
        conf = probs[m].mean(); acc = labels[m].mean()
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)


def write_weekly_report(log_dir="data/logs", art_dir="artifacts/weekly", tp_pips=10.0, sl_pips=5.0):
    import os
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from pathlib import Path

    Path(art_dir).mkdir(parents=True, exist_ok=True)
    sig_p = os.path.join(log_dir, "signals.parquet")
    ord_p = os.path.join(log_dir, "orders.parquet")
    trd_p = os.path.join(log_dir, "trades.parquet")

    df_sig = pd.read_parquet(sig_p) if os.path.exists(sig_p) else pd.DataFrame()
    df_ord = pd.read_parquet(ord_p) if os.path.exists(ord_p) else pd.DataFrame()
    df_trd = pd.read_parquet(trd_p) if os.path.exists(trd_p) else pd.DataFrame()

    n_trades = len(df_trd)
    win = df_trd.get("tp_hit")
    win = (win.astype(bool) if win is not None else None)
    pnl_pips = df_trd.get("pnl_pips")
    pnl_pips = (pnl_pips.astype(float) if pnl_pips is not None else None)

    W = float(win.mean()) if win is not None and n_trades else 0.0
    PF = (float(pnl_pips[pnl_pips>0].sum()) / max(1e-9, float(-pnl_pips[pnl_pips<0].sum()))) if pnl_pips is not None and n_trades else 0.0
    EV = float(pnl_pips.mean()) if pnl_pips is not None and n_trades else 0.0
    eq = pnl_pips.cumsum() if pnl_pips is not None else None
    MaxDD = float((eq.cummax() - eq).max()) if eq is not None and n_trades else 0.0

    if "ev_pred_pips" in df_ord.columns:
        EV_pred = float(df_ord["ev_pred_pips"].mean())
    elif "ev_pred" in df_ord.columns:
        EV_pred = float(df_ord["ev_pred"].mean())
    elif not df_sig.empty and "p_cal" in df_sig.columns:
        EV_pred = float((df_sig["p_cal"] * tp_pips - (1 - df_sig["p_cal"]) * sl_pips).mean())
    else:
        EV_pred = float('nan')
    EV_real = EV
    EV_gap = float(abs(EV_pred - EV_real)) if (not pd.isna(EV_pred)) else float('nan')

    ECE = float('nan')
    if not df_sig.empty and not df_trd.empty and "p_cal" in df_sig.columns and "tp_hit" in df_trd.columns:
        try:
            n = min(len(df_sig), len(df_trd))
            probs = df_sig.tail(n)["p_cal"].to_numpy()
            labels = df_trd.tail(n)["tp_hit"].astype(int).to_numpy()
            ECE = _ece(probs, labels, n_bins=10)
        except Exception:
            pass

    reject_share = {}
    if not df_ord.empty and "action" in df_ord.columns and "reason" in df_ord.columns:
        rej = df_ord[df_ord["action"] == "reject"]
        if not rej.empty:
            c = rej["reason"].value_counts(dropna=False)
            t = float(c.sum())
            for k, v in c.items():
                reject_share[str(k)] = round(float(v) / max(1.0, t) * 100.0, 2)

    ab_share = {}
    if not df_trd.empty and "variant" in df_trd.columns:
        c = df_trd["variant"].value_counts()
        t = float(c.sum())
        for k, v in c.items():
            ab_share[str(k)] = round(float(v) / max(1.0, t) * 100.0, 2)

    def _md_header(title):
        from datetime import datetime
        return f"# {title}\n\n生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    md = []
    md.append(_md_header("Weekly Trading Report"))
    md.append("## 稼働KPI\n")
    md.append(f"- 本数: **{n_trades}**  \n- 勝率 W: **{W:.2%}**  \n- PF: **{PF:.2f}**  \n- EV[pips/回]: **{EV:.3f}**  \n- MaxDD[pips]: **{MaxDD:.1f}**\n")
    md.append("## TCA / 校正\n")
    md.append(f"- EV_pred 平均[pips]: **{(EV_pred if EV_pred==EV_pred else 'NaN')}**  \n- EV_real 平均[pips]: **{EV:.3f}**  \n- |EV_pred−EV_real|: **{(EV_gap if EV_gap==EV_gap else 'NaN')}**  \n- ECE(10bins): **{(ECE if ECE==ECE else 'NaN')}**\n")
    md.append("## Reject内訳\n")
    if reject_share:
        md.append("```\n" + "\n".join([f"{k}: {v:.2f}%" for k, v in reject_share.items()]) + "\n```\n")
    else:
        md.append("- （該当なし）\n")
    md.append("## A/B配分\n")
    if ab_share:
        md.append("```\n" + "\n".join([f"{k}: {v:.2f}%" for k, v in ab_share.items()]) + "\n```\n")
    else:
        md.append("- （該当なし）\n")

    out = os.path.join(art_dir, f"report_{pd.Timestamp.now().strftime('%Y%m%d')}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"[weekly] wrote {out}")


def main():
    cfg = {
        "tca": yaml.safe_load(open("configs/tca.yml", "r", encoding="utf-8")).get("tca", {}),
    }
    # 1) Online calibration (if labels available)
    if os.path.exists(SIGNALS_LOG):
        df = pd.read_parquet(SIGNALS_LOG)
        if {"p_raw", "y"}.issubset(df.columns) and len(df) >= 500:
            cal = OnlineCalibrator(method="isotonic")
            cal.fit_and_eval(df.tail(5000))
            path = cal.save_artifact()
            print("[calibration] saved:", path)
        else:
            print("[calibration] skipped (labels not present or too few rows)")
    else:
        print("[calibration] no signals log")

    # 2) TCA
    path = fit_from_logs({})
    if path:
        print("[tca] model saved:", path)
    else:
        print("[tca] skipped")

    # 3) A/B evaluation
    try:
        mgr = ABManager()
        w, m = mgr.evaluate_and_promote()
        print("[ab] winner:", w, m)
    except Exception as e:
        print("[ab] skipped:", e)

    # 4) Weekly report
    try:
        write_weekly_report()
    except Exception as e:
        print("[weekly] report skipped:", e)


if __name__ == "__main__":
    main()
