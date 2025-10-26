import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Trade & Risk Filters", page_icon="🛡️", layout="wide")
st.title("Trade & Risk Filters")
try:
    from src.ui.safety_badge import render_safety_badge
    render_safety_badge()
except Exception:
    pass

# ATR/kSL/kTP panel (always-on). If df is not loaded yet, will fallback later using data_path.
_atr_panel_added = False
try:
    from src.ui.atr_panel import render_atr_panel
    # We'll render once after df is loaded below; temporarily show with default path to avoid delay
    render_atr_panel(data_path=st.session_state.get("default_data_path", "data/USDJPY_15m.csv"))
    _atr_panel_added = True
except Exception:
    pass

DATA_PATH_DEFAULT = Path("data/USDJPY_15m.csv")
EVENTS_PATH_DEFAULT = Path("data/events.csv")  # time列（JST/naive想定）

with st.sidebar:
    st.subheader("Inputs")
    data_path = st.text_input("Market CSV path", str(DATA_PATH_DEFAULT))
    events_path = st.text_input("Events CSV path (optional)", str(EVENTS_PATH_DEFAULT))
    max_spread = st.number_input("Max spread (pips)", min_value=0.0, value=1.0, step=0.1)
    min_atr = st.number_input("Min ATR (optional)", min_value=0.0, value=0.0, step=0.1)
    max_atr = st.number_input("Max ATR (optional)", min_value=0.0, value=2.0, step=0.1)
    news_before = st.number_input("News window before (min)", min_value=0, value=10, step=5)
    news_after = st.number_input("News window after (min)", min_value=0, value=10, step=5)


def safe_read_csv(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"CSV read failed: {e}")
        return None


df = safe_read_csv(Path(data_path))
if df is None:
    st.info("マーケットCSVが見つかりません（例: data/USDJPY_15m.csv）。存在しなくてもページは落ちません。")
    st.stop()

# Re-render panel with in-memory df (more accurate and faster for subsequent UI)
try:
    if not _atr_panel_added:
        render_atr_panel(df=df)
except Exception:
    pass

# 軽バリデーション（Pandera）
try:
    from src.core.schema import validate_market_df

    _, schema_msgs = validate_market_df(df)
    if schema_msgs:
        with st.expander("⚠️ データスキーマ警告（上位10件）"):
            for m in schema_msgs:
                st.warning(str(m))
except Exception:
    # Pandera未インストールなどでもページは落とさない
    pass

rows = len(df)
st.write(f"Rows: {rows}")
preview_cols = [c for c in ["time", "close", "spread_pips", "atr"] if c in df.columns]
st.dataframe(df[preview_cols].tail(10) if preview_cols else df.tail(10), use_container_width=True)

# フィルタ適用（列が無ければ自動No-Op）
from src.core.ta import add_atr_if_missing, add_spread_pips_if_missing
from src.policy.filters import apply_spread_filter, apply_atr_filter, apply_news_window_filter
from src.policy.df_guard import enforce_env_guard_df

# 欠損列を必要時だけ自動算出（存在すれば何もしない）
out = add_spread_pips_if_missing(df.copy(), pip_size=0.01)
out = add_atr_if_missing(out)

out = apply_spread_filter(out, max_spread_pips=float(max_spread))
out = apply_atr_filter(out, min_atr=float(min_atr) if min_atr > 0 else None, max_atr=float(max_atr) if max_atr > 0 else None)

events_df = safe_read_csv(Path(events_path)) or pd.DataFrame(columns=["time"])
out = apply_news_window_filter(out, events=events_df, minutes_before=int(news_before), minutes_after=int(news_after))

# 最後に環境ガード（MODE/KILL）
out = enforce_env_guard_df(out)

ok = int(out.get("trade_ok", pd.Series([], dtype=bool)).sum()) if "trade_ok" in out.columns else 0
ng = rows - ok
c1, c2 = st.columns(2)
c1.metric("Allow (trade_ok=True)", ok)
c2.metric("Blocked", ng)

st.subheader("Sample")
show_blocked_only = st.checkbox("Blocked（trade_ok=False）のみ表示", value=False)
sample_df = out.tail(50)
if show_blocked_only and "trade_ok" in sample_df.columns:
    sample_df = sample_df[sample_df["trade_ok"] == False]  # noqa: E712
st.dataframe(sample_df, use_container_width=True)

st.caption("※ 必要列が無ければ各フィルタは自動で何もしません（安全No-Op）。")

with st.expander("Reasons summary"):
    # reason -> deny_reason -> gate_reason の優先順で集計
    reason_col = next((c for c in ["reason", "deny_reason", "gate_reason"] if c in out.columns), None)
    if reason_col is not None:
        vc = (
            out[reason_col]
            .astype(str)
            .replace({"": None, "nan": None})
            .dropna()
            .value_counts()
            .head(20)
        )
        st.write(vc)
        try:
            st.bar_chart(vc)
        except Exception:
            pass
    else:
        st.info("理由列（reason/deny_reason/gate_reason）がまだありません。フィルタや環境ガードをONにすると表示されます。")

st.divider()
st.subheader("Quality gate mini dashboard (enabled when thresholds YAML is non-empty)")

def _load_thresholds_yaml() -> dict:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    p = Path("config/patterns_quality_thresholds.yml")
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data or {}
    except Exception:
        return {}

thr_map = _load_thresholds_yaml()
if not thr_map:
    st.info("品質ゲートは未設定です（No-Op）。config/patterns_quality_thresholds.yml に値を入れると有効になります。")
else:
    # logs/executions.parquet から直近1日/7日の block/quality-block を集計
    logs_p = Path("logs/executions.parquet")
    if not logs_p.exists():
        st.info("logs/executions.parquet が見つかりません。運用ログが出始めると集計されます。")
    else:
        try:
            logs = pd.read_parquet(logs_p)
            if "time" in logs.columns:
                logs["time"] = pd.to_datetime(logs["time"], errors="coerce")
            now = pd.Timestamp.now()
            def _summary(days: int) -> dict:
                dfw = logs.copy()
                if "time" in dfw.columns:
                    dfw = dfw[dfw["time"] >= (now - pd.Timedelta(days=days))]
                total = len(dfw)
                blocked = int((dfw.get("trade_ok", pd.Series([], dtype=bool)).astype(bool) == False).sum()) if total else 0  # noqa: E712
                qual = 0
                if total:
                    col = next((c for c in ["reason", "deny_reason", "gate_reason"] if c in dfw.columns), None)
                    if col:
                        qual = int(dfw[col].astype(str).str.contains("pattern_quality<thr", case=False, na=False).sum())
                return {"total": total, "blocked": blocked, "blocked_quality": qual, "block_rate": (blocked/total if total else 0.0)}

            s1 = _summary(1)
            s7 = _summary(7)
            c1, c2, c3 = st.columns(3)
            c1.metric("1d: blocked", s1["blocked"], help=f"quality={s1['blocked_quality']} / total={s1['total']}")
            c2.metric("7d: blocked", s7["blocked"], help=f"quality={s7['blocked_quality']} / total={s7['total']}")
            c3.metric("1d block rate", f"{s1['block_rate']:.2%}")
            st.caption("quality=pattern_quality<thr で弾かれた件数。しきい値は config/patterns_quality_thresholds.yml を参照。")
        except Exception as e:
            st.error(f"ログ集計に失敗しました: {e}")
