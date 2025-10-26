import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

from src.monitoring.observability import (
    load_logs,
    daily_allow_block,
    top_reasons,
    export_monthly_report,
    export_periodic_report,
    export_this_week,
    export_last_week,
)
from src.policy.reasons_normalize import load_reason_map, normalize_reasons_df
from src.core.session import add_session
from src.monitoring.health import assess_health, HealthThresholds
from src.core.safety import KILL_FILE, ensure_flags_dir
import json
import glob
import os
import numpy as np
import plotly.graph_objects as go
from src.core.drift import window_drift
from src.ui.cache import file_mtime, read_csv_cached, read_parquet_cached


st.set_page_config(page_title="Observability", page_icon="📊", layout="wide")
st.title("Observability")
try:
    from src.ui.safety_badge import render_safety_badge
    render_safety_badge()
except Exception:
    pass

# ATR/kSL/kTP panel (always-on, safe no-op) - uses default data path if available
try:
    from src.ui.atr_panel import render_atr_panel
    render_atr_panel(data_path=st.session_state.get("default_data_path", "data/USDJPY_15m.csv"))
except Exception:
    pass

with st.expander("Config change alert (latest two reports)"):
    try:
        meta_paths = sorted(glob.glob("reports/monthly_*/metadata.json")) + sorted(
            glob.glob("reports/periodic_*/metadata.json")
        )
        if len(meta_paths) >= 2:
            a = json.loads(Path(meta_paths[-1]).read_text(encoding="utf-8"))
            b = json.loads(Path(meta_paths[-2]).read_text(encoding="utf-8"))
            ah = a.get("config_hashes", {})
            bh = b.get("config_hashes", {})
            if ah != bh:
                st.warning("⚠️ Config hashes changed between the last two reports.")
                st.code({"latest": ah, "previous": bh})
            else:
                st.success("Config hashes unchanged between the last two reports.")
        else:
            st.caption("Create at least two reports to enable config diff.")
    except Exception as e:
        st.info(f"metadata.json comparison skipped: {e}")

with st.sidebar:
    st.subheader("Sources")
    exec_path = st.text_input("Executions log (parquet/jsonl/csv)", "logs/executions.parquet")
    trades_path = st.text_input("Trades log (optional)", "logs/trades.parquet")
    days = st.number_input("Lookback days", min_value=1, value=14, step=1)

    st.subheader("Export options")
    out_root = st.text_input("Report root", "reports")
    as_of = st.date_input("As of (month will be used)", value=datetime.now().date())
    
    st.subheader("Date range (optional)")
    today = datetime.now().date()
    start_date = st.date_input("Start date", value=today - timedelta(days=int(days)))
    end_date = st.date_input("End date", value=today)
    apply_range = st.checkbox("Apply range filter to charts", value=False)

    st.subheader("Reason normalization")
    reasons_map_path = st.text_input("Reasons map YAML path", "config/reasons_map.yml")
    use_normalized = st.checkbox("Normalize reasons in charts/export (if map exists)", value=False)

paths = [Path(exec_path)]
if trades_path:
    paths.append(Path(trades_path))

df = load_logs(paths)
if df.empty:
    st.info("ログが見つかりません（logs/executions.parquet 等）。存在しなくてもページは落ちません。")
    st.stop()

since = datetime.now() - timedelta(days=int(days))
if "time" in df.columns:
    df = df[pd.to_datetime(df["time"], errors="coerce") >= since]

if apply_range:
    m = (
        pd.to_datetime(df["time"], errors="coerce") >= pd.to_datetime(start_date)
    ) & (
        pd.to_datetime(df["time"], errors="coerce")
        <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    )
    df = df[m]

st.subheader("Daily Allow / Block")
ts = daily_allow_block(df)
if ts.empty:
    st.info("対象期間にデータがありません。")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(ts.set_index("day")[["allow", "block"]])
    with c2:
        st.line_chart(ts.set_index("day")[ ["block_rate"] ])
    st.dataframe(ts.tail(30), use_container_width=True)

st.subheader("Top Reasons")
# 正規化（存在時のみ）
mapping = load_reason_map(Path(reasons_map_path)) if use_normalized else {}
df_for_view = normalize_reasons_df(df, mapping=mapping) if mapping else df
reasons = top_reasons(df_for_view, top_n=10, since=since, normalized=bool(mapping))
if reasons.empty:
    st.info("理由の集計対象がありません。")
else:
    st.write(reasons)
    try:
        st.bar_chart(reasons)
    except Exception:
        pass

st.subheader("Export")
if st.button("💾 Export monthly report (CSV)"):
    try:
        paths = [Path(exec_path)] + ([Path(trades_path)] if trades_path else [])
        out_dir = export_monthly_report(
            df_for_view if mapping else df,
            as_of=datetime.combine(as_of, datetime.min.time()),
            out_root=Path(out_root),
            sources=paths,
            reasons_map_path=Path(reasons_map_path) if mapping else None,
        )
        st.success(f"Exported to {out_dir} (metadata.json written)")
    except Exception as e:
        st.error(f"Export failed: {e}")

if st.button("📅 Export period report (CSV)"):
    try:
        paths = [Path(exec_path)] + ([Path(trades_path)] if trades_path else [])
        out_dir = export_periodic_report(
            df_for_view if mapping else df,  # （どちらでもOKだが、normalizedありならそれを優先）
            start=start_date,
            end=end_date,
            out_root=Path(out_root),
            sources=paths,
            reasons_map_path=Path(reasons_map_path) if mapping else None,
        )
        st.success(f"Exported to {out_dir} (metadata.json written)")
    except Exception as e:
        st.error(f"Export failed: {e}")

# 生成済みのCSVがあれば簡易ダウンロード
with st.expander("Download recent report CSVs"):
    try:
        csvs = sorted(glob.glob("reports/**/daily.csv", recursive=True)) + sorted(
            glob.glob("reports/**/top_reasons*.csv", recursive=True)
        )
        if not csvs:
            st.caption("まだレポートCSVがありません。上の Export から生成できます。")
        else:
            for p in csvs[-10:]:
                try:
                    st.download_button("Download " + p, data=Path(p).read_bytes(), file_name=Path(p).name)
                except Exception:
                    pass
    except Exception:
        pass

c1, c2 = st.columns(2)
if c1.button("🗓️ Export THIS WEEK"):
    try:
        paths = [Path(exec_path)] + ([Path(trades_path)] if trades_path else [])
        out_dir = export_this_week(
            df_for_view if mapping else df,
            out_root=Path(out_root),
            sources=paths,
            reasons_map_path=Path(reasons_map_path) if mapping else None,
        )
        st.success(f"Exported to {out_dir}")
    except Exception as e:
        st.error(f"Export failed: {e}")

if c2.button("🗓️ Export LAST WEEK"):
    try:
        paths = [Path(exec_path)] + ([Path(trades_path)] if trades_path else [])
        out_dir = export_last_week(
            df_for_view if mapping else df,
            out_root=Path(out_root),
            sources=paths,
            reasons_map_path=Path(reasons_map_path) if mapping else None,
        )
        st.success(f"Exported to {out_dir}")
    except Exception as e:
        st.error(f"Export failed: {e}")

    c3 = st.checkbox("Show session block-rate by day (optional)", value=False)
    if c3:
        try:
            df_s = add_session(df_for_view)
            if "session" in df_s.columns and "trade_ok" in df_s.columns and "time" in df_s.columns:
                tmp = df_s.copy()
                tmp["day"] = pd.to_datetime(tmp["time"], errors="coerce").dt.date
                g = tmp.groupby(["day", "session"])
                stats = g["trade_ok"].agg(["count", lambda s: (s.astype(bool) == False).sum()]).reset_index()  # noqa: E712
                stats.columns = ["day", "session", "total", "blocked"]
                stats["block_rate"] = stats.apply(lambda r: (r["blocked"] / r["total"]) if r["total"] else 0.0, axis=1)
                st.subheader("Block rate by session (recent)")
                st.dataframe(stats.sort_values(["day", "session"]).tail(50), use_container_width=True)
            else:
                st.info("必要な列（time/trade_ok）がありません。")
        except Exception as e:
            st.info(f"session view skipped: {e}")

# ---- Drift quick check (PSI/JS) ----
with st.expander("Drift quick check (PSI/JS)"):
    st.caption("任意のCSV/Parquet から列を選び、参照窓 vs 現在窓で PSI/JS を計算します。")

    src_path = st.text_input("Source file (csv/parquet)", "data/USDJPY_15m.csv")
    drift_col = st.text_input("Column", "close")

    c1, c2, c3 = st.columns(3)
    with c1:
        ref_n = st.number_input("ref_n", min_value=10, value=2000, step=10)
    with c2:
        cur_n = st.number_input("cur_n", min_value=10, value=500, step=10)
    with c3:
        bins = st.number_input("bins", min_value=5, value=20, step=1)

    d1, d2 = st.columns(2)
    with d1:
        th_psi = st.number_input("Max PSI (optional)", min_value=0.0, value=float(os.getenv("HEALTH_MAX_DRIFT_PSI", "0.5")))
    with d2:
        th_js = st.number_input("Max JS  (optional)", min_value=0.0, value=float(os.getenv("HEALTH_MAX_DRIFT_JS", "0.3")))

    # 追加: アーティファクト保存のオプション
    with st.container():
        persist = st.checkbox("Save drift artifacts (summary.json + hist.csv)", value=False)
        note = st.text_input("Optional note", "")

    # ファイル読込とドリフト計算をキャッシュ
    @st.cache_data(show_spinner=False)
    def _get_mtime(path: str) -> float | None:
        return file_mtime(path)

    @st.cache_data(show_spinner=False)
    def _cached_read_any(path: str, mtime: float | None) -> pd.DataFrame | None:
        try:
            if not path:
                return None
            p = Path(path)
            if not p.exists():
                return None
            if p.suffix.lower() == ".parquet":
                return read_parquet_cached(path, mtime)
            return read_csv_cached(path, mtime)
        except Exception:
            return None

    @st.cache_data(show_spinner=False)
    def _compute_drift_and_hist(path: str, mtime: float | None, col: str, ref_n: int, cur_n: int, bins: int):
        df = _cached_read_any(path, mtime)
        if df is None or col not in df.columns:
            raise ValueError("source/column not available")
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < (ref_n + cur_n):
            raise ValueError(f"series length {len(s)} < ref_n+cur_n {ref_n+cur_n}")
        # PSI/JS（window_driftは dict を返す）
        metrics = window_drift(s, ref_n=ref_n, cur_n=cur_n, bins=bins)
        psi_val = float(metrics.get("psi", float("nan")))
        js_val = float(metrics.get("js", float("nan")))
        # 窓の切り方（window_driftと整合：末尾 cur_n ／ その直前 ref_n）
        s_ref = s.iloc[-(ref_n + cur_n):-cur_n]
        s_cur = s.iloc[-cur_n:]
        # 共有ビン
        vmin = float(np.nanmin([s_ref.min(), s_cur.min()]))
        vmax = float(np.nanmax([s_ref.max(), s_cur.max()]))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            # 退避（ゼロ幅回避）
            vmin, vmax = float(s.min()), float(s.max() + 1e-9)
        edges = np.linspace(vmin, vmax, int(bins) + 1)
        ref_cnt, _ = np.histogram(s_ref, bins=edges)
        cur_cnt, _ = np.histogram(s_cur, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2.0
        hist_df = pd.DataFrame({
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": centers,
            "ref_count": ref_cnt,
            "cur_count": cur_cnt,
        })
        return psi_val, js_val, hist_df

    if st.button("▶ Compute drift"):
        try:
            mtime = _get_mtime(src_path)
            df_src = _cached_read_any(src_path, mtime)
            if df_src is None or drift_col not in (df_src.columns if df_src is not None else []):
                st.warning("入力ファイルが見つからないか、指定列が存在しません。")
            else:
                psi_val, js_val, hist_df = _compute_drift_and_hist(src_path, mtime, drift_col, int(ref_n), int(cur_n), int(bins))
                c1, c2 = st.columns(2)
                c1.metric("PSI", f"{psi_val:.4f}")
                c2.metric("JS", f"{js_val:.4f}")
                ok = True
                if th_psi > 0:
                    ok = ok and (psi_val <= th_psi)
                if th_js > 0:
                    ok = ok and (js_val <= th_js)
                if ok:
                    st.success("✅ Thresholds satisfied.")
                else:
                    st.error("⚠️ Drift thresholds exceeded.")

                # 軽量ヒストグラム（共有ビン）
                try:
                    fig = go.Figure()
                    fig.add_bar(x=hist_df["bin_center"], y=hist_df["ref_count"], name="ref")
                    fig.add_bar(x=hist_df["bin_center"], y=hist_df["cur_count"], name="cur", opacity=0.7)
                    fig.update_layout(barmode="overlay", title="Histogram: ref vs cur", xaxis_title=drift_col)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

                # 保存（任意）
                if persist:
                    out_dir = Path("reports") / f"drift_{datetime.now().strftime('%Y%m%d%H%M')}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    # summary.json
                    summary = {
                        "generated_at": datetime.now().isoformat(),
                        "source": src_path,
                        "column": drift_col,
                        "ref_n": int(ref_n),
                        "cur_n": int(cur_n),
                        "bins": int(bins),
                        "psi": float(psi_val),
                        "js": float(js_val),
                        "thresholds": {
                            "psi": float(th_psi) if th_psi > 0 else None,
                            "js": float(th_js) if th_js > 0 else None,
                        },
                        "note": note or "",
                    }
                    (out_dir / "summary.json").write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    # hist.csv
                    hist_df.to_csv(out_dir / "hist.csv", index=False, encoding="utf-8")
                    st.success(f"Saved: {out_dir}/summary.json, hist.csv")
        except Exception as e:
            st.info(f"Drift check skipped: {e}")

st.markdown("---")

st.subheader("🚦 Health gate (run with current settings)")

lg1, lg2 = st.columns([3, 2])
with lg1:
    logs_text = st.text_input(
        "Log files (space-separated)", "logs/executions.parquet logs/trades.parquet"
    )
with lg2:
    lookback_days = st.number_input("Lookback days", min_value=1, value=1, step=1)

h1, h2 = st.columns(2)
with h1:
    max_block = st.number_input(
        "Max block_rate",
        min_value=0.0,
        max_value=1.0,
        value=float(os.getenv("HEALTH_MAX_BLOCK_RATE", "0.8")),
    )
with h2:
    max_unknown = st.number_input(
        "Max unknown_ratio",
        min_value=0.0,
        max_value=1.0,
        value=float(os.getenv("HEALTH_MAX_UNKNOWN_RATIO", "0.3")),
    )

auto_kill = st.checkbox("If FAILED, apply Kill Switch (file)", value=False)
if st.button("Run health gate with current settings"):
    try:
        th = HealthThresholds(
            lookback_days=int(lookback_days),
            max_block_rate=float(max_block),
            max_unknown_ratio=float(max_unknown),
            # ドリフトのUI値もシグネチャに合わせる場合はここに渡す（未設定ならNoneのまま）
        )
        logs_paths = [Path(p) for p in logs_text.split() if p]
        kwargs = {
            "drift_source": Path(src_path) if src_path else None,
            "drift_column": drift_col,
            "drift_ref_n": int(ref_n),
            "drift_cur_n": int(cur_n),
            "drift_bins": int(bins),
        }
        # try extended signature -> fallback to legacy
        try:
            res = assess_health(logs_paths, th, Path("config/reasons_map.yml"), **kwargs)
        except TypeError:
            res = assess_health(logs_paths, th, Path("config/reasons_map.yml"))

        st.json(res)
        if res.get("ok"):
            st.success("✅ Health gate OK")
        else:
            st.error("⚠️ Health gate FAILED")
            if auto_kill:
                try:
                    ensure_flags_dir()
                    KILL_FILE.parent.mkdir(parents=True, exist_ok=True)
                    KILL_FILE.write_text("kill", encoding="utf-8")
                    st.warning("🛑 Kill Switch file created (flags/kill.switch). Trading disabled.")
                except Exception as ee:
                    st.info(f"Kill Switch failed: {ee}")

        out_dir = Path("reports") / f"health_{datetime.now().strftime('%Y%m%d%H%M')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "inputs": {
                "logs": [str(p) for p in logs_paths],
                "drift_source": src_path,
                "drift_column": drift_col,
                "ref_n": int(ref_n),
                "cur_n": int(cur_n),
                "bins": int(bins),
                "thresholds": {
                    "max_block_rate": float(max_block),
                    "max_unknown_ratio": float(max_unknown),
                    "max_drift_psi": float(th_psi) if th_psi > 0 else None,
                    "max_drift_js": float(th_js) if th_js > 0 else None,
                },
            },
            "result": res,
        }
        (out_dir / "health.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        st.success(f"Saved: {out_dir}/health.json")
    except Exception as e:
        st.error(f"Health gate failed: {e}")

# --- Gate failure records (from logs/gate_failures.jsonl) ---
with st.expander("Gate failures (recent)"):
    try:
        import json
        from pathlib import Path as _P

        path = _P("logs") / "gate_failures.jsonl"
        rows = []
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        if not rows:
            st.caption("No gate failure records yet.")
        else:
            df_fail = pd.DataFrame(rows)
            st.dataframe(
                df_fail[[c for c in ["ts", "reason", "hint", "details"] if c in df_fail.columns]].tail(100),
                use_container_width=True,
            )
    except Exception as e:
        st.info(f"Gate failure view skipped: {e}")
