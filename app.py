# -*- coding: utf-8 -*-
# Streamlit FX Auto Lines - 完全版 + News Shading + Flag/Pennant + H&S + Ghost Projection
# 黒背景・重要度別ニュースウィンドウ赤影・ソフト抑制・自動ライン/パターン/EV/ブレイク確率・手動再学習

import os, math, json, subprocess, sys, pathlib, re, warnings
from datetime import timedelta
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from dataclasses import dataclass
import pytz, joblib

warnings.filterwarnings("ignore")

# ---------------- Page Config ----------------
st.set_page_config(page_title="FX 自動ライン描画 - 完全版", page_icon="📈", layout="wide")
JST = pytz.timezone("Asia/Tokyo")

# ---------------- ダーク配色 ----------------
COLOR_BG = "#0b0f14"
COLOR_GRID = "#263238"
COLOR_TEXT = "#e0f2f1"
COLOR_LEVEL = "#00e5ff"            # 水平線
COLOR_TREND = "#ffa000"            # トレンドライン
COLOR_CH_UP = "#ff6e6e"            # チャネル上
COLOR_CH_DN = "#64b5f6"            # チャネル下
COLOR_CANDLE_UP_BODY = "#26a69a"
COLOR_CANDLE_UP_EDGE = "#66fff9"
COLOR_CANDLE_DN_BODY = "#ef5350"
COLOR_CANDLE_DN_EDGE = "#ff8a80"

# ---------------- Intraday制約クランプ ----------------
INTRADAY_SET = {"1m","2m","5m","15m","30m","60m","90m"}
MAX_PERIOD_BY_INTERVAL = {
    "1m":"7d","2m":"60d","5m":"60d","15m":"60d","30m":"60d",
    "60m":"730d","90m":"730d"
}
def clamp_period_for_interval(period: str, interval: str) -> str:
    if interval not in INTRADAY_SET: return period
    maxp = MAX_PERIOD_BY_INTERVAL.get(interval)
    if not maxp: return period
    def to_days(p: str) -> int:
        p = p.strip().lower()
        if p.endswith("d"):  return int(p[:-1])
        if p.endswith("mo"): return int(p[:-2]) * 30
        if p.endswith("y"):  return int(p[:-1]) * 365
        return 999999
    return maxp if to_days(period) > to_days(maxp) else period

# ---------------- Utils ----------------
def ensure_jst_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    return df.copy().set_index(idx.tz_convert(JST))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def in_sessions(ts: pd.Timestamp) -> str:
    h = ts.hour
    if 9 <= h < 15: return "Tokyo"
    if 16<= h < 24: return "London"
    if h>=22 or h<5: return "NY"
    return "Other"

def pip_value(pair="USDJPY"):
    return 0.01  # USDJPY想定

def _select_first(values) -> str:
    return list(dict.fromkeys(map(str, values)))[0]

def normalize_ohlcv(df: pd.DataFrame, symbol: str | None) -> pd.DataFrame:
    """yfinanceの単層/多層列を open/high/low/close に統一"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    if not isinstance(df.columns, pd.MultiIndex):
        df = df.rename(columns=lambda c: str(c).strip()).rename(columns=str.lower)
        need = {"open","high","low","close"}
        if not need.issubset(set(df.columns)):
            colmap = {}
            for want in ["Open","High","Low","Close","Adj Close","Volume"]:
                cand = [c for c in df.columns if c.lower().startswith(want.lower())]
                if cand: colmap[cand[0]] = want.lower()
            df = df.rename(columns=colmap)
            if not need.issubset(set(df.columns)):
                raise ValueError(f"必須列が見つかりません: {need - set(df.columns)}")
        return df

    # MultiIndex対応
    cols = df.columns
    lvl0, lvl1 = set(map(str, cols.get_level_values(0))), set(map(str, cols.get_level_values(1)))
    fields = {"Open","High","Low","Close","Adj Close","Volume"}
    if fields & lvl0 and not (fields & lvl1):
        df = df.swaplevel(0,1,axis=1)
        cols = df.columns

    if fields & set(map(str, cols.get_level_values(1))):
        tickers = list(dict.fromkeys(map(str, cols.get_level_values(0))))
        pick_ticker = symbol if (symbol and symbol in tickers) else _select_first(cols.get_level_values(0))
        sub = df.xs(pick_ticker, axis=1, level=0)
        if isinstance(sub.columns, pd.MultiIndex):
            try: sub = sub.droplevel(0, axis=1)
            except Exception: sub.columns = ["_".join(map(str,c)) for c in sub.columns]
        want = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in sub.columns]
        if not {"Open","High","Low","Close"}.issubset(set(want)):
            def _find(name):
                for c in sub.columns:
                    if str(c).lower()==name.lower(): return c
                for c in sub.columns:
                    if name.lower() in str(c).lower(): return c
                return None
            pick = {k:_find(k) for k in ["Open","High","Low","Close","Adj Close","Volume"]}
            if not all(pick[k] for k in ["Open","High","Low","Close"]):
                raise ValueError("OHLCV列の抽出に失敗（異例のMultiIndex）。")
            want = [pick[k] for k in ["Open","High","Low","Close"] if pick[k] is not None]
            if pick.get("Adj Close"): want.append(pick["Adj Close"])
            if pick.get("Volume"):    want.append(pick["Volume"])
        out = sub[want].copy()
        out.columns = [str(c).lower() for c in out.columns]
        return out

    df_flat = df.copy()
    df_flat.columns = ["_".join(map(str, c)) for c in df_flat.columns]
    mapping = {}
    for fld in ["Open","High","Low","Close","Adj Close","Volume"]:
        cand = [c for c in df_flat.columns if c.split("_")[-1].lower()==fld.lower() or fld.lower() in c.lower()]
        if cand: mapping[fld]=cand[0]
    if not {"Open","High","Low","Close"}.issubset(mapping.keys()):
        raise ValueError("OHLCV列の抽出に失敗（未知の列構成）。")
    cols_ordered = [mapping[k] for k in ["Open","High","Low","Close"] if k in mapping]
    if "Adj Close" in mapping: cols_ordered.append(mapping["Adj Close"])
    if "Volume" in mapping:    cols_ordered.append(mapping["Volume"])
    out = df_flat[cols_ordered].copy()
    out.columns = [c.split("_")[-1].lower() for c in out.columns]
    return out

# ---------------- サイドバー ----------------
st.sidebar.title("設定")
symbol = st.sidebar.text_input("ティッカー（USDJPY）", value="JPY=X", help="yfinanceでUSDJPYは 'JPY=X'")
period_raw = st.sidebar.selectbox("取得期間", ["7d","14d","30d","60d","90d","180d","1y"], index=2)
interval = st.sidebar.selectbox("足種", ["5m","15m","30m","60m","1d"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("自動更新")
auto_refresh = st.sidebar.checkbox("自動で再取得（ページ再読み込み）", value=True)
refresh_secs = st.sidebar.slider("更新間隔（秒）", 30, 600, 60, help="15分足は60〜180秒が目安")
try:
    from streamlit_autorefresh import st_autorefresh
    if auto_refresh:
        st_autorefresh(interval=refresh_secs * 1000, limit=None, key="fx_autorefresh")
except Exception:
    if auto_refresh:
        from streamlit.components.v1 import html
        html(f"""<script>setTimeout(function(){{window.location.reload();}}, {int(refresh_secs*1000)});</script>""", height=0)

st.sidebar.markdown("---")
st.sidebar.subheader("シグナル条件")
signal_mode = st.sidebar.selectbox(
    "種別を選択",
    ["水平線ブレイク(終値)", "トレンドラインブレイク(終値)", "チャネル上抜け/下抜け(終値)", "リテスト指値(水平線)"],
    index=0
)
retest_wait_k_base = st.sidebar.slider("リテスト待機本数K", 3, 30, 10)
st.sidebar.caption("ブレイク後、K本以内にライン/バンドへ戻ったかで『リテストあり/なし』を判定（指数0〜1も算出）")

st.sidebar.markdown("---")
st.sidebar.subheader("極値検出（スイング）")
look = st.sidebar.slider("左右の窓幅", 3, 15, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("水平サポレジ（クラスタ）")
eps = st.sidebar.number_input("DBSCAN eps（価格）", value=0.08, step=0.01)
min_samples = st.sidebar.slider("min_samples", 3, 12, 4)

st.sidebar.markdown("---")
st.sidebar.subheader("トレンド＆チャネル")
reg_lookback = st.sidebar.slider("回帰に使う直近本数", 50, 400, 150)
chan_k = st.sidebar.slider("チャネル幅（σの倍率）", 0.5, 3.0, 2.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("判定バッファ")
touch_buffer = st.sidebar.number_input("接触バッファ（価格）", value=0.05, step=0.01)
break_buffer_base = st.sidebar.number_input("ブレイクバッファ（価格）", value=0.05, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("重要度スコアの重み")
w_touch = st.sidebar.slider("接触回数", 0.0, 1.0, 0.30, 0.05)
w_recent = st.sidebar.slider("直近距離（近いほど↑）", 0.0, 1.0, 0.30, 0.05)
w_session = st.sidebar.slider("時間帯（主要セッションで↑）", 0.0, 1.0, 0.20, 0.05)
w_vol = st.sidebar.slider("ボラ（ATR）", 0.0, 1.0, 0.20, 0.05)
w_sum = max(1e-9, w_touch + w_recent + w_session + w_vol)
w_touch, w_recent, w_session, w_vol = [w/w_sum for w in (w_touch, w_recent, w_session, w_vol)]

# ---------- ニュース・指標フィルタ & 赤影 ----------
st.sidebar.markdown("---")
st.sidebar.subheader("ニュース・指標フィルタ / 赤影")
news_file = st.sidebar.file_uploader("ニュースCSVをアップロード（任意）", type=["csv"])
st.sidebar.caption("受理列: time/timestamp/datetime または date+time、importance[, title]（JST推奨）")

# フィルタ方式
news_filter_mode = st.sidebar.radio(
    "フィルタ方式",
    ["一律±分", "重要度別（赤影と同じ）"],
    index=1, horizontal=True
)
news_win = st.sidebar.slider("一律±分（上を選んだときのみ使用）", 0, 120, 30)
news_imp_min = st.sidebar.slider("重要度しきい値 (>=)", 1, 5, 3)

# 重要度→±分マッピング（赤影/重要度別フィルタで使用）
st.sidebar.caption("重要度別ウィンドウ（左右±分）")
map_5 = st.sidebar.number_input("★5 → ±分", value=90, step=5)
map_4 = st.sidebar.number_input("★4 → ±分", value=30, step=5)
map_3 = st.sidebar.number_input("★3 → ±分", value=20, step=5)
map_2 = st.sidebar.number_input("★2 → ±分", value=0, step=5)
map_1 = st.sidebar.number_input("★1 → ±分", value=0, step=5)
use_news_shade = st.sidebar.checkbox("チャートに赤影を重ねて表示", value=True)

# ハード/ソフト抑制
apply_news_filter = st.sidebar.checkbox("ハード抑制（窓内のシグナル無効化）", value=True)
use_soft_suppress = st.sidebar.checkbox("ソフト抑制（窓内だけ判定を厳しめに）", value=True)
soft_break_add = st.sidebar.number_input("ソフト: ブレイクバッファ 追加", value=0.02, step=0.01, format="%.2f")
soft_K_add = st.sidebar.slider("ソフト: K 追加", 0, 10, 4)

# ---------------- バックテスト ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("バックテスト")
fwd_n = st.sidebar.slider("ブレイク後 N 本（損益判定）", 5, 120, 20)
spread_pips = st.sidebar.number_input("想定スプレッド（pips）", value=0.5, step=0.1)
run_bt = st.sidebar.button("▶ バックテストを実行")

# ---------------- ブレイク確率 / EV ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("ブレイク確率（今から）")
show_break_prob = st.sidebar.checkbox("今からの水平線ブレイク確率を表示", value=True)
break_prob_topk = st.sidebar.slider("表示する上位ライン数", 1, 20, 10)
break_prob_h = st.sidebar.slider("先読みH（参考表示）", 3, 50, 12)
prob_model_path = st.sidebar.text_input("モデルパス", value="models/break_model.joblib")

st.sidebar.markdown("---")
st.sidebar.subheader("期待値ランキング（確率 × 期待pips）")
show_ev_rank = st.sidebar.checkbox("水平線の期待値ランキングを表示", value=True)
ev_level_min_samples = st.sidebar.slider("レベル別の最低サンプル数", 1, 20, 3)

# ---------- 手動再学習ボタン ----------
st.sidebar.markdown("---")
st.sidebar.subheader("モデルの手動再学習")
proj_dir = str(pathlib.Path(__file__).resolve().parent)
train_script = "ai_train_break.py"
model_path = "models/break_model.joblib"
colA, colB = st.sidebar.columns([1,1])
with colA:
    retrain_now = st.button("再学習を実行", type="primary")
with colB:
    show_log = st.checkbox("ログを表示", value=True)

def _format_ts(ts: float) -> str:
    return pd.Timestamp.fromtimestamp(ts, tz=JST).strftime("%Y-%m-%d %H:%M:%S %Z")

def run_retrain(script_path: str, workdir: str) -> tuple[bool, str]:
    py = sys.executable
    try:
        proc = subprocess.run(
            [py, script_path],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60*20
        )
        ok = proc.returncode == 0
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return ok, log
    except Exception as e:
        return False, f"Exception: {e}"

if retrain_now:
    with st.spinner("学習スクリプトを実行中..."):
        ok, log = run_retrain(train_script, proj_dir)
    model_file = pathlib.Path(proj_dir, model_path)
    if model_file.exists():
        ts = model_file.stat().st_mtime
        st.success(f"学習完了：{model_path} を更新（{_format_ts(ts)}）")
        try:
            st.cache_resource.clear()
        except Exception:
            pass
    else:
        st.error("学習は終了しましたが、モデルファイルが見つかりません。保存パスをご確認ください。")
    if show_log:
        st.subheader("再学習ログ")
        st.code((log or "").strip()[:200000], language="bash")
else:
    mf = pathlib.Path(proj_dir, model_path)
    if mf.exists():
        st.caption(f"現在のモデル更新: {_format_ts(mf.stat().st_mtime)}")
    else:
        st.caption("モデル未作成（先に再学習を実行してください）")

# ---------------- データ取得 ----------------
@st.cache_data(show_spinner=False, ttl=60)
def load_data(sym: str, period: str, interval: str) -> pd.DataFrame:
    adj_period = clamp_period_for_interval(period, interval)
    try:
        df = yf.Ticker(sym).history(period=adj_period, interval=interval, auto_adjust=False)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        df = yf.download(sym, period=adj_period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return normalize_ohlcv(df, sym)

with st.spinner("データ取得中..."):
    df_raw = load_data(symbol, period_raw, interval)
if df_raw.empty:
    st.error("データが取得できませんでした。ティッカー/期間/足を確認してください。")
    st.stop()
df = ensure_jst_index(df_raw)
pv = pip_value("USDJPY")

# ---------------- 極値 & レベル ----------------
def swing_pivots(df: pd.DataFrame, look: int):
    highs = df["high"].rolling(look, center=True).max()
    lows  = df["low"].rolling(look, center=True).min()
    pivot_high = df[(df["high"] == highs)].dropna(subset=["high"])
    pivot_low  = df[(df["low"]  == lows )].dropna(subset=["low"])
    return pivot_high, pivot_low

def horizontal_levels(pivot_high: pd.DataFrame, pivot_low: pd.DataFrame, eps: float, min_samples: int):
    prices = np.r_[pivot_high["high"].values, pivot_low["low"].values].reshape(-1,1)
    if len(prices) == 0: return []
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(prices).labels_
    levels = []
    for lab in set(labels) - {-1}:
        lv = prices[labels==lab].mean()
        levels.append(float(lv))
    return sorted(levels)

pivot_high, pivot_low = swing_pivots(df, look)
levels = horizontal_levels(pivot_high, pivot_low, eps=eps, min_samples=min_samples)

# ---------------- 回帰トレンド & チャネル ----------------
def regression_trend(df: pd.DataFrame, lookback: int, use="low"):
    sub = df.tail(lookback)
    y = sub[use].values
    x = np.arange(len(y))
    if len(x) < 2: return None
    m, b = np.polyfit(x, y, 1)
    t0, t1 = sub.index[0], sub.index[-1]
    y0, y1 = b, m*(len(x)-1) + b
    resid = y - (m*x + b)
    sigma = float(np.std(resid))
    return dict(x0=t0, y0=y0, x1=t1, y1=y1, slope=m, intercept=b, sigma=sigma, n=len(x))

trend = regression_trend(df, reg_lookback, use="low")

# ---------------- レベル重要度スコア ----------------
def compute_level_scores(df: pd.DataFrame, levels: list, touch_buffer: float,
                         w_touch: float, w_recent: float, w_session: float, w_vol: float) -> pd.DataFrame:
    if not levels:
        return pd.DataFrame(columns=["level","score","touches","session_ratio"])
    _atr = atr(df, 14)
    atr_recent = float(_atr.iloc[-1]) if not _atr.empty else 0.0
    rec_close = float(df["close"].iloc[-1])

    rows=[]
    for lv in levels:
        touch_mask = ((df["low"] <= lv) & (df["high"] >= lv)) | (df["close"].sub(lv).abs() <= touch_buffer)
        touches = int(touch_mask.sum())
        dist = abs(rec_close - lv)
        near = 1.0 / (dist + 1e-6)
        if touches > 0:
            ts_idx = df.index[touch_mask]
            sess_hits = sum(1 for ts in ts_idx if in_sessions(ts) in ("Tokyo","London","NY"))
            session_ratio = sess_hits / touches
        else:
            session_ratio = 0.0
        atr_norm = atr_recent / max(1e-6, rec_close)
        rows.append(dict(level=float(lv), touches=touches, near=near, session_ratio=session_ratio, atr_norm=atr_norm))
    df_sc = pd.DataFrame(rows)
    for col in ["touches","near","session_ratio","atr_norm"]:
        colmin, colmax = df_sc[col].min(), df_sc[col].max()
        df_sc[col+"_n"] = 0.0 if math.isclose(colmin, colmax) else (df_sc[col]-colmin)/(colmax-colmin)
    df_sc["score"] = (
        w_touch*df_sc["touches_n"] + w_recent*df_sc["near_n"] +
        w_session*df_sc["session_ratio_n"] + w_vol*df_sc["atr_norm_n"]
    ) * 100.0
    return df_sc.sort_values("score", ascending=False)

score_df = compute_level_scores(df, levels, touch_buffer, w_touch, w_recent, w_session, w_vol)

# ---------------- ニュースCSV（堅牢パーサ） ----------------
def parse_news_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame(columns=["time","importance","title"])
    try:
        df = pd.read_csv(file, dtype=str)
    except Exception:
        file.seek(0); df = pd.read_csv(file, dtype=str, header=None)
    df = df.astype(str).fillna("")
    def norm(s: str) -> str:
        s = s.strip().replace("\u3000"," "); s = s.lower()
        return re.sub(r"[\s\-_/()]", "", s)
    TIME_CANDS = {"time","timestamp","datetime","date_time","timejst","datetimejst","日時","日付時刻","発表時刻","発表時間","時刻","時間","date","when"}
    DATE_ONLY = {"date","日付","発表日"}
    CLOCK_ONLY= {"time","時刻","発表時刻","発表時間","時間"}
    IMP_CANDS  = {"importance","重要度","impact","rank","priority","優先度","star","stars"}
    TITLE_CANDS= {"title","イベント","指標名","headline","event","name","内容","subject"}
    orig_cols = list(df.columns)
    norm_cols = [norm(c) for c in orig_cols]
    col_map = dict(zip(norm_cols, orig_cols))
    time_col=imp_col=title_col=None
    for nc in norm_cols:
        if nc in {norm(x) for x in TIME_CANDS} and nc not in {norm(x) for x in DATE_ONLY}:
            time_col = col_map[nc]; break
    if time_col is None:
        date_col = None; clock_col=None
        for nc in norm_cols:
            if nc in {norm(x) for x in DATE_ONLY}: date_col = col_map[nc]
            if nc in {norm(x) for x in CLOCK_ONLY}: clock_col = col_map[nc]
        if date_col and clock_col:
            df["_tmp_time"] = (df[date_col].astype(str).str.strip()+" "+df[clock_col].astype(str).str.strip()).str.strip()
            time_col = "_tmp_time"
    if time_col is None:
        best_col=None; best_ok=-1
        for c in df.columns:
            tryconv = pd.to_datetime(df[c], utc=True, errors="coerce", infer_datetime_format=True)
            ok = tryconv.notna().sum()
            if ok > best_ok and ok>0:
                best_ok = ok; best_col = c
        if best_col is not None: time_col = best_col
    for nc in norm_cols:
        if nc in {norm(x) for x in IMP_CANDS}:
            imp_col = col_map[nc]; break
    if imp_col is None:
        best_col=None; best_numeric=-1
        for c in df.columns:
            if c == time_col: continue
            s = pd.to_numeric(df[c], errors="coerce")
            numeric_ok = s.notna().sum()
            if numeric_ok > best_numeric and numeric_ok>0:
                best_numeric = numeric_ok; best_col=c
        if best_col is not None: imp_col = best_col
    for nc in norm_cols:
        if nc in {norm(x) for x in TITLE_CANDS}:
            title_col = col_map[nc]; break
    if title_col is None:
        cand = [c for c in df.columns if c not in {time_col, imp_col}]
        title_col = cand[0] if cand else None

    if time_col is None or imp_col is None:
        raise ValueError("ニュースCSVに 'time'（または date+time）と 'importance' が必要です。")

    def parse_dt_series(s: pd.Series) -> pd.Series:
        dt = pd.to_datetime(s, utc=True, errors="coerce", infer_datetime_format=True)
        bad = dt.isna()
        if bad.any():
            fmt_list = ["%Y-%m-%d %H:%M","%Y/%m/%d %H:%M","%Y-%m-%d %H:%M:%S","%Y/%m/%d %H:%M:%S"]
            raw = s[bad].astype(str).str.strip()
            fixed = pd.Series([pd.NaT]*len(raw), index=raw.index)
            for fmt in fmt_list:
                try:
                    parsed = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
                    fixed = fixed.fillna(parsed)
                except Exception:
                    pass
            dt.loc[bad] = fixed
        bad = dt.isna()
        if bad.any():
            raw = s[bad].astype(str).str.strip()
            def numparse(x):
                try:
                    v = float(x)
                    if v > 10_000_000_000: return pd.to_datetime(v, unit="ms", utc=True)
                    return pd.to_datetime(v, unit="s", utc=True)
                except Exception:
                    return pd.NaT
            dt.loc[bad] = raw.apply(numparse)
        return dt

    dt_utc = parse_dt_series(df[time_col])
    dt_utc = pd.to_datetime(dt_utc, utc=True, errors="coerce")
    if dt_utc.notna().sum() == 0:
        raise ValueError("ニュースCSVの日時を解釈できませんでした。")
    dt_jst = dt_utc.dt.tz_convert(JST)
    imp = pd.to_numeric(df[imp_col], errors="coerce").fillna(0).astype(int)
    ttl = df[title_col] if (title_col in df.columns) else ""
    out = pd.DataFrame({"time": dt_jst, "importance": imp, "title": ttl}).dropna(subset=["time"])
    return out.sort_values("time").reset_index(drop=True)

news_df = parse_news_csv(news_file)

# ---- 重要度別ウィンドウ生成 & 赤影描画ユーティリティ ----
def build_event_windows(events_df: pd.DataFrame, imp_threshold: int, mapping: dict[int,int]) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=["start","end","importance","title"])
    rows=[]
    for _, r in events_df.iterrows():
        imp = int(r["importance"])
        if imp < imp_threshold: 
            continue
        minutes = mapping.get(imp, 0)
        start = r["time"] - timedelta(minutes=minutes)
        end   = r["time"] + timedelta(minutes=minutes)
        rows.append({"start": start, "end": end, "importance": imp, "title": r.get("title", "")})
    windows = pd.DataFrame(rows)
    return windows.sort_values("start").reset_index(drop=True)

def is_suppressed(ts: pd.Timestamp, windows_df: pd.DataFrame) -> bool:
    if windows_df.empty: return False
    return bool(((windows_df["start"] <= ts) & (ts <= windows_df["end"])).any())

def add_news_shading_to_fig(fig: go.Figure, windows_df: pd.DataFrame) -> go.Figure:
    if windows_df.empty: return fig
    color_map = {5:"rgba(255,0,0,0.18)", 4:"rgba(255,0,0,0.12)", 3:"rgba(255,0,0,0.08)"}
    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    for _, r in windows_df.iterrows():
        col = color_map.get(int(r["importance"]), "rgba(255,0,0,0.08)")
        shapes.append(dict(type="rect", xref="x", x0=r["start"], x1=r["end"],
                           yref="paper", y0=0, y1=1, fillcolor=col, line=dict(width=0), layer="below"))
    fig.update_layout(shapes=shapes)
    return fig

# マッピング辞書
imp_map = {5:int(map_5), 4:int(map_4), 3:int(map_3), 2:int(map_2), 1:int(map_1)}
windows_df = build_event_windows(news_df, imp_threshold=news_imp_min, mapping=imp_map) if news_df is not None else pd.DataFrame()

# ---------------- パターン検出（Triangle / Rectangle / Double / Flag / Pennant / H&S） ----------------
@dataclass
class Pattern:
    kind: str
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    params: dict
    quality: float
    direction_bias: str

def _fit_line(xs, ys):
    if len(xs) < 2: 
        return None
    m,b = np.polyfit(xs, ys, 1)
    return m,b

def detect_triangle(df, piv_high, piv_low, lookback=200, min_touches=2, max_future_cross=100):
    sub = df.tail(lookback)
    hi_idx = piv_high.index.intersection(sub.index)
    lo_idx = piv_low.index.intersection(sub.index)
    idx_map = {t:i for i,t in enumerate(sub.index)}
    xs_h = np.array([idx_map[t] for t in hi_idx], dtype=float)
    ys_h = sub.loc[hi_idx, "high"].values if len(hi_idx)>0 else np.array([])
    xs_l = np.array([idx_map[t] for t in lo_idx], dtype=float)
    ys_l = sub.loc[lo_idx, "low"].values if len(lo_idx)>0 else np.array([])
    if len(xs_h) < min_touches or len(xs_l) < min_touches:
        return []
    uh = _fit_line(xs_h, ys_h); ll = _fit_line(xs_l, ys_l)
    if not uh or not ll: return []
    (m1,b1),(m2,b2) = uh, ll
    out=[]
    if abs(m1-m2) > 1e-9:
        x_star = (b2-b1)/(m1-m2)
        x_now  = len(sub)-1
        if x_now <= x_star <= x_now + max_future_cross:
            y_u = m1*x_now + b1; y_l = m2*x_now + b2
            thickness = max(0.0, y_u - y_l)
            slope_up = (m2>0 and abs(m1) < abs(m2)*0.6)
            slope_dn = (m1<0 and abs(m2) < abs(m1)*0.6)
            kind = "triangle_sym"
            if slope_up: kind = "triangle_up"
            if slope_dn: kind = "triangle_dn"
            quality = 60.0 + (10.0 if thickness>0 else 0.0)
            out.append(Pattern(kind, sub.index[0], sub.index[-1],
                               dict(upper=(m1,b1), lower=(m2,b2), thickness=thickness,
                                    sub_start=sub.index[0], sub_end=sub.index[-1]),
                               quality, "neutral"))
    return out

def detect_rectangle(df, lookback=200, min_touches=3, tol=0.05):
    sub = df.tail(lookback)
    if len(sub) < 20: return []
    up = sub["high"].rolling(max(5, lookback//3)).max().iloc[-1]
    dn = sub["low"].rolling(max(5, lookback//3)).min().iloc[-1]
    height = up - dn
    if height <= 0: return []
    mask_up = ((sub["high"] >= up - tol) & (sub["low"] <= up + tol))
    mask_dn = ((sub["high"] >= dn - tol) & (sub["low"] <= dn + tol))
    touches = int((mask_up | mask_dn).sum())
    if touches < min_touches: return []
    q = 55 + min(30, (touches-min_touches)*5)
    return [Pattern("rectangle", sub.index[0], sub.index[-1],
                    dict(upper=float(up), lower=float(dn), height=float(height),
                         sub_start=sub.index[0], sub_end=sub.index[-1]),
                    q, "neutral")]

def detect_double_top_bottom(df, piv_high, piv_low, lookback=200, tol=0.1, min_gap=5):
    sub = df.tail(lookback)
    out=[]
    highs = piv_high.index.intersection(sub.index).sort_values()
    for i in range(len(highs)-1):
        p1,p2 = highs[i], highs[i+1]
        if (sub.index.get_loc(p2) - sub.index.get_loc(p1)) < min_gap: continue
        if abs(df.loc[p1,"high"] - df.loc[p2,"high"]) <= tol:
            neck = float(df.loc[p1:p2, "low"].min())
            topv = float(df.loc[p1,"high"])
            out.append(Pattern("double_top", p1, p2, dict(top=topv, neck=neck), 60.0, "down"))
    lows = piv_low.index.intersection(sub.index).sort_values()
    for i in range(len(lows)-1):
        p1,p2 = lows[i], lows[i+1]
        if (sub.index.get_loc(p2) - sub.index.get_loc(p1)) < min_gap: continue
        if abs(df.loc[p1,"low"] - df.loc[p2,"low"]) <= tol:
            neck = float(df.loc[p1:p2, "high"].max())
            botv = float(df.loc[p1,"low"])
            out.append(Pattern("double_bottom", p1, p2, dict(bottom=botv, neck=neck), 60.0, "up"))
    return out

def detect_flag_pennant(df, lookback=220, Npush=30, min_flag_bars=8, max_flag_bars=40, sigma_k=1.0, pole_min_atr=2.0):
    if len(df) < Npush + max_flag_bars + 5: return []
    sub = df.tail(lookback)
    cons = sub.tail(max_flag_bars)
    pole = sub.iloc[-(max_flag_bars+Npush):-max_flag_bars]
    if len(pole) < 5 or len(cons) < min_flag_bars: return []

    pole_len = float(pole["close"].iloc[-1] - pole["close"].iloc[0])
    pole_dir = "up" if pole_len > 0 else "down"
    pole_abs = abs(pole_len)

    atr_now = float(atr(sub, 14).iloc[-1]) if len(sub)>=14 else 0.0
    if atr_now <= 0: return []
    if pole_abs < pole_min_atr * atr_now:
        return []

    y = cons["close"].values
    x = np.arange(len(y))
    m, b = np.polyfit(x, y, 1)
    resid = y - (m*x + b)
    sigma = float(np.std(resid))
    if sigma <= 0: return []
    x_now = len(x)-1
    y_mid_now = m*x_now + b
    up_now = y_mid_now + sigma_k*sigma
    dn_now = y_mid_now - sigma_k*sigma

    kind = "pennant" if abs(m) < 1e-4 else ("flag_up" if m>0 else "flag_dn")
    direction_bias = "up" if pole_dir=="up" else "down"
    quality = 65.0
    quality += min(20.0, pole_abs/atr_now*3.0)
    quality += min(10.0, (1.0/(sigma/atr_now+1e-6)))

    return [Pattern(
        kind=kind,
        t_start=cons.index[0], t_end=cons.index[-1],
        params=dict(slope=m, intercept=b, sigma=sigma, band_k=float(sigma_k),
                    sub_start=cons.index[0], sub_end=cons.index[-1],
                    pole=float(pole_len), pole_abs=float(pole_abs),
                    upper_now=float(up_now), lower_now=float(dn_now)),
        quality=float(quality),
        direction_bias=direction_bias
    )]

def detect_head_shoulders(df, piv_high, piv_low, lookback=260, tol=0.003):
    out=[]
    sub = df.tail(lookback)

    # トップ型（H&S）
    hs = piv_high.index.intersection(sub.index).sort_values()
    if len(hs) >= 3:
        for i in range(len(hs)-2):
            s1, h, s2 = hs[i], hs[i+1], hs[i+2]
            v1, vh, v2 = float(df.loc[s1,"high"]), float(df.loc[h,"high"]), float(df.loc[s2,"high"])
            if vh > v1*(1+tol) and vh > v2*(1+tol) and abs(v1 - v2) <= max( tol*vh, 0.1 ):
                left_valley  = float(df.loc[s1:h, "low"].min())
                right_valley = float(df.loc[h:s2, "low"].min())
                neck = (left_valley + right_valley)/2.0
                q = 70.0 + min(15.0, (vh - max(v1,v2))/max(1e-6, atr(sub,14).iloc[-1]) )
                out.append(Pattern(
                    kind="head_shoulders",
                    t_start=s1, t_end=s2,
                    params=dict(head=vh, left=v1, right=v2, neck=neck),
                    quality=float(q), direction_bias="down"
                ))

    # ボトム型（逆H&S）
    ls = piv_low.index.intersection(sub.index).sort_values()
    if len(ls) >= 3:
        for i in range(len(ls)-2):
            s1, h, s2 = ls[i], ls[i+1], ls[i+2]
            v1, vh, v2 = float(df.loc[s1,"low"]), float(df.loc[h,"low"]), float(df.loc[s2,"low"])
            if vh < v1*(1-tol) and vh < v2*(1-tol) and abs(v1 - v2) <= max( tol*max(v1,v2), 0.1 ):
                left_peak  = float(df.loc[s1:h, "high"].max())
                right_peak = float(df.loc[h:s2, "high"].max())
                neck = (left_peak + right_peak)/2.0
                q = 70.0 + min(15.0, (min(v1,v2) - vh)/max(1e-6, atr(sub,14).iloc[-1]) )
                out.append(Pattern(
                    kind="inverse_head_shoulders",
                    t_start=s1, t_end=s2,
                    params=dict(head=vh, left=v1, right=v2, neck=neck),
                    quality=float(q), direction_bias="up"
                ))
    return out

def measured_targets(p: Pattern):
    """パターン別ターゲット（測定値）"""
    if p.kind.startswith("triangle"):
        t = float(p.params.get("thickness", 0.0))
        return dict(up=+t, down=-t)
    if p.kind=="rectangle":
        h = float(p.params.get("height", 0.0))
        return dict(up=+h, down=-h)
    if p.kind in ("double_top","double_bottom"):
        neck = p.params.get("neck")
        ref  = p.params.get("top", p.params.get("bottom", None))
        if neck is not None and ref is not None:
            h = abs(float(ref) - float(neck))
            return dict(up=+h, down=-h)
    if p.kind in ("flag_up","flag_dn","pennant"):
        pole = float(p.params.get("pole_abs", 0.0))
        return dict(up=+pole, down=-pole)
    if p.kind in ("head_shoulders","inverse_head_shoulders"):
        neck = float(p.params.get("neck", 0.0))
        head = float(p.params.get("head", 0.0))
        h = abs(head - neck)
        return dict(up=+h, down=-h)
    return dict(up=0.0, down=0.0)

# ---- パターン検出設定（サイドバー）----
st.sidebar.markdown("---")
st.sidebar.subheader("パターン検出")
enable_tri    = st.sidebar.checkbox("トライアングル（対称/上昇/下降）", True)
enable_rect   = st.sidebar.checkbox("レクタングル（ボックス）", True)
enable_double = st.sidebar.checkbox("ダブルトップ / ダブルボトム", True)
enable_flag   = st.sidebar.checkbox("フラッグ / ペナント", True)
enable_hs     = st.sidebar.checkbox("ヘッド＆ショルダーズ（逆含む）", True)

pat_lookback = st.sidebar.slider("検出対象の直近本数", 100, 600, 220, 20)
pat_min_touches = st.sidebar.slider("最小接触回数（線へのタッチ数）", 2, 5, 2)
pat_tol_price = st.sidebar.number_input("価格許容誤差（ダブル/ボックス）", value=0.10, step=0.05)

# Flag/Pennant パラメータ
st.sidebar.caption("— フラッグ/ペナント 設定 —")
flag_Npush = st.sidebar.slider("旗竿推定 Npush（本）", 10, 60, 30, 2)
flag_min_bars = st.sidebar.slider("調整ゾーン最小本数", 6, 30, 10, 2)
flag_max_bars = st.sidebar.slider("調整ゾーン最大本数", 10, 80, 40, 2)
flag_sigma_k = st.sidebar.slider("σバンド倍率（境界）", 0.5, 3.0, 1.0, 0.5)
flag_pole_min_atr = st.sidebar.slider("旗竿の最小強度（ATR倍）", 1.0, 5.0, 2.0, 0.5)

# H&S パラメータ
st.sidebar.caption("— ヘッド＆ショルダーズ 設定 —")
hs_tol = st.sidebar.slider("肩の高さ許容（比率）", 0.001, 0.02, 0.003, 0.001)

# ---- パターン検出実行 ----
patterns = []
try:
    if enable_tri:
        patterns += detect_triangle(df, pivot_high, pivot_low,
                                    lookback=pat_lookback, min_touches=pat_min_touches, max_future_cross=100)
    if enable_rect:
        patterns += detect_rectangle(df, lookback=pat_lookback, min_touches=pat_min_touches, tol=pat_tol_price)
    if enable_double:
        patterns += detect_double_top_bottom(df, pivot_high, pivot_low,
                                             lookback=pat_lookback, tol=pat_tol_price, min_gap=look+2)
    if enable_flag:
        patterns += detect_flag_pennant(df, lookback=pat_lookback, Npush=flag_Npush,
                                        min_flag_bars=flag_min_bars, max_flag_bars=flag_max_bars,
                                        sigma_k=flag_sigma_k, pole_min_atr=flag_pole_min_atr)
    if enable_hs:
        patterns += detect_head_shoulders(df, pivot_high, pivot_low, lookback=pat_lookback, tol=hs_tol)
except Exception as e:
    st.error(f"パターン検出中にエラー: {e}")
    patterns = []

# ---------------- モデル読み込み（TTL付きキャッシュ） ----------------
@st.cache_resource(show_spinner=False, ttl=600)
def _load_break_model_cached(path: str):
    return joblib.load(path)

def load_break_model(path: str):
    try:
        obj = _load_break_model_cached(path)
        return obj["model"], obj.get("Xcols", None), obj.get("meta", {})
    except Exception:
        return None, None, None

def session_onehot_feat(ts):
    h = ts.hour
    return (1.0 if 9<=h<15 else 0.0,
            1.0 if 16<=h<24 else 0.0,
            1.0 if h>=22 or h<5 else 0.0)

def make_features_for_level(df, ts, level, dir_sign, touch_buffer, trend_look=150):
    i = len(df)-1
    c = float(df["close"].iloc[i])
    a = atr(df, 14).fillna(0.0).iloc[i]
    lb = max(0, i-trend_look+1)
    y = df["close"].iloc[lb:i+1].values
    slope = 0.0
    if len(y) >= 2:
        m, b = np.polyfit(np.arange(len(y)), y, 1); slope = float(m)
    dist = abs(c - level)
    near = 1.0/(dist+1e-6)
    Ntouch=200
    sub = df.iloc[max(0, i-Ntouch):i+1]
    touches = int((((sub["low"]<=level)&(sub["high"]>=level)) | (sub["close"].sub(level).abs()<=touch_buffer)).sum())
    last_touch_bar = 0
    for k in range(i, max(-1, i-Ntouch), -1):
        if (df["low"].iloc[k] <= level <= df["high"].iloc[k]) or (abs(df["close"].iloc[k]-level) <= touch_buffer):
            last_touch_bar = i-k; break
    tokyo,london,ny = session_onehot_feat(ts)
    sign = 1 if (dir_sign==1 and c>=level) or (dir_sign==-1 and c<=level) else -1
    return [dir_sign, dist, sign, near, a/max(1e-6,c), slope, touches, last_touch_bar, tokyo, london, ny]

# ====================== チャート描画 ======================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="Price",
    increasing_line_color=COLOR_CANDLE_UP_EDGE, increasing_fillcolor=COLOR_CANDLE_UP_BODY,
    decreasing_line_color=COLOR_CANDLE_DN_EDGE, decreasing_fillcolor=COLOR_CANDLE_DN_BODY
))

# 水平線
score_df = compute_level_scores(df, levels, touch_buffer, w_touch, w_recent, w_session, w_vol)
if not score_df.empty:
    sc_min, sc_max = score_df["score"].min(), score_df["score"].max()
    rng = max(1e-9, (sc_max - sc_min))
    for _, r in score_df.iterrows():
        alpha = 0.25 if math.isclose(sc_min, sc_max) else 0.20 + 0.65 * (r["score"]-sc_min)/rng
        fig.add_hline(y=r["level"], opacity=float(alpha), line_color=COLOR_LEVEL, line_width=1.5)
else:
    for lv in levels:
        fig.add_hline(y=lv, opacity=0.35, line_color=COLOR_LEVEL, line_width=1.2)

# トレンド＆チャネル
trend = regression_trend(df, reg_lookback, use="low")
if trend:
    fig.add_shape(type="line", x0=trend["x0"], y0=trend["y0"], x1=trend["x1"], y1=trend["y1"],
                  line=dict(width=2.4, color=COLOR_TREND))
    if trend["sigma"] > 0:
        x0, x1 = trend["x0"], trend["x1"]
        y0_u = trend["y0"] + chan_k*trend["sigma"]; y1_u = trend["y1"] + chan_k*trend["sigma"]
        y0_l = trend["y0"] - chan_k*trend["sigma"]; y1_l = trend["y1"] - chan_k*trend["sigma"]
        fig.add_shape(type="line", x0=x0, y0=y0_u, x1=x1, y1=y1_u,
                      line=dict(width=1.6, color=COLOR_CH_UP, dash="dot"))
        fig.add_shape(type="line", x0=x0, y0=y0_l, x1=x1, y1=y1_l,
                      line=dict(width=1.6, color=COLOR_CH_DN, dash="dot"))

# パターン描画
def _draw_triangle(fig, p: Pattern):
    sub = df.loc[p.params["sub_start"] : p.params["sub_end"]]
    x = np.arange(len(sub))
    (m1,b1) = p.params["upper"]; (m2,b2) = p.params["lower"]
    y_u = m1*x + b1; y_l = m2*x + b2
    fig.add_trace(go.Scatter(x=sub.index, y=y_u, mode="lines", name=f"{p.kind}-upper",
                             line=dict(color="#ffd54f", width=2, dash="dash"), showlegend=False))
    fig.add_trace(go.Scatter(x=sub.index, y=y_l, mode="lines", name=f"{p.kind}-lower",
                             line=dict(color="#4fc3f7", width=2, dash="dash"), showlegend=False))

def _draw_rectangle(fig, p: Pattern):
    sub = df.loc[p.params["sub_start"] : p.params["sub_end"]]
    up = p.params["upper"]; dn = p.params["lower"]
    fig.add_hrect(y0=dn, y1=up, x0=sub.index[0], x1=sub.index[-1],
                  line_width=0, fillcolor="rgba(0,150,136,0.12)")
    fig.add_hline(y=up, line=dict(color="#ffd54f", width=2, dash="dot"))
    fig.add_hline(y=dn, line=dict(color="#4fc3f7", width=2, dash="dot"))

def _draw_double(fig, p: Pattern):
    if p.kind=="double_top":
        top = p.params["top"]; neck = p.params["neck"]
        fig.add_hline(y=top, line=dict(color="#ffd54f", width=2))
        fig.add_hline(y=neck, line=dict(color="#90caf9", width=2, dash="dot"))
    else:
        bot = p.params["bottom"]; neck = p.params["neck"]
        fig.add_hline(y=bot, line=dict(color="#4fc3f7", width=2))
        fig.add_hline(y=neck, line=dict(color="#90caf9", width=2, dash="dot"))

def _draw_flag(fig, p: Pattern):
    sub = df.loc[p.params["sub_start"] : p.params["sub_end"]]
    x = np.arange(len(sub))
    m = p.params["slope"]; b = p.params["intercept"]; s = p.params["sigma"]; k = p.params["band_k"]
    y_mid = m*x + b
    y_up = y_mid + k*s
    y_dn = y_mid - k*s
    fig.add_trace(go.Scatter(x=sub.index, y=y_up, mode="lines", name="flag_up",
                             line=dict(color="#ffcc80", width=2, dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=sub.index, y=y_dn, mode="lines", name="flag_dn",
                             line=dict(color="#80d8ff", width=2, dash="dot"), showlegend=False))

def _draw_hs(fig, p: Pattern):
    neck = p.params.get("neck", None)
    if neck is not None:
        fig.add_hline(y=float(neck), line=dict(color="#f48fb1", width=2, dash="dash"))
    fig.add_annotation(x=p.t_end, y=p.params.get("head", float(df['close'].iloc[-1])),
                       text=("H&S" if p.kind=="head_shoulders" else "Inv H&S"),
                       showarrow=False, font=dict(color="#f06292"))

for p in patterns:
    if p.kind.startswith("triangle"):
        _draw_triangle(fig, p)
    elif p.kind=="rectangle":
        _draw_rectangle(fig, p)
    elif p.kind in ("double_top","double_bottom"):
        _draw_double(fig, p)
    elif p.kind in ("flag_up","flag_dn","pennant"):
        _draw_flag(fig, p)
    elif p.kind in ("head_shoulders","inverse_head_shoulders"):
        _draw_hs(fig, p)

# ---- 赤影（重要度別ウィンドウ）を重ねる ----
if use_news_shade and not windows_df.empty:
    fig = add_news_shading_to_fig(fig, windows_df)

# --- 疑似チャート投影（ゴースト）コントロール ---
st.sidebar.markdown("---")
st.sidebar.subheader("疑似チャート投影（ゴースト）")
enable_ghost = st.sidebar.checkbox("パターンから将来パスを薄く重ねる", value=False)
ghost_h = st.sidebar.slider("投影本数（バー）", 10, 120, 40, help="何本先まで薄く描くか")
ghost_mode = st.sidebar.selectbox("方法", ["EV直線", "ボラ扇形(平均)"], index=0)
ghost_alpha = st.sidebar.slider("透明度", 0.05, 0.6, 0.18)
ghost_fan_k = st.sidebar.slider("扇形の幅k（σ係数）", 0.5, 3.0, 1.5, 0.5)
ghost_sims = st.sidebar.slider("ランダムウォーク本数（任意）", 0, 300, 0)

# --- ゴーストのための補助関数 ---
def _pattern_levels_for_prob(df, p: Pattern):
    upper_level = None; lower_level = None
    if p.kind.startswith("triangle"):
        subp = df.loc[p.params["sub_start"]:p.params["sub_end"]]
        m1,b1 = p.params["upper"]; m2,b2 = p.params["lower"]
        x_now  = len(subp)-1
        upper_level = float(m1*x_now + b1)
        lower_level = float(m2*x_now + b2)
    elif p.kind=="rectangle":
        upper_level = float(p.params["upper"]); lower_level = float(p.params["lower"])
    elif p.kind=="double_top":
        lower_level = float(p.params["neck"])
    elif p.kind=="double_bottom":
        upper_level = float(p.params["neck"])
    elif p.kind in ("flag_up","flag_dn","pennant"):
        upper_level = float(p.params["upper_now"]); lower_level = float(p.params["lower_now"])
    elif p.kind=="head_shoulders":
        lower_level = float(p.params["neck"])
    elif p.kind=="inverse_head_shoulders":
        upper_level = float(p.params["neck"])
    return upper_level, lower_level

def _pick_best_pattern(patterns: list[Pattern]) -> Pattern | None:
    if not patterns: return None
    pats = sorted(patterns, key=lambda x: (x.quality, x.t_end), reverse=True)
    return pats[0]

def _ghost_path_ev(df, P_up, P_dn, tgt_up_px, tgt_dn_px, h: int):
    w_up = float(P_up) if not np.isnan(P_up) else 0.5
    w_dn = float(P_dn) if not np.isnan(P_dn) else 0.5
    if (w_up + w_dn) <= 1e-9:
        w_up = w_dn = 0.5
    else:
        s = w_up + w_dn
        w_up /= s; w_dn /= s
    c0 = float(df["close"].iloc[-1])
    xs = np.arange(h+1)
    up_line = c0 + (tgt_up_px - c0) * (xs / h)
    dn_line = c0 + (tgt_dn_px - c0) * (xs / h)
    y = w_up*up_line + w_dn*dn_line
    return xs, y

def _ghost_cone_from_atr(df, h: int, k: float):
    c0 = float(df["close"].iloc[-1])
    _atr = atr(df, 14).iloc[-1] if len(df)>=14 else (df["high"].iloc[-1]-df["low"].iloc[-1])
    xs = np.arange(h+1)
    sigma = float(_atr)
    spread = k * sigma * np.sqrt(np.maximum(0, xs))
    mid = np.full_like(xs, c0, dtype=float)
    return xs, mid, mid+spread, mid-spread

def _rand_walks(df, h: int, n: int, drift_line: np.ndarray | None):
    if n <= 0: return []
    c0 = float(df["close"].iloc[-1])
    _atr = atr(df, 14).iloc[-1] if len(df)>=14 else (df["high"].iloc[-1]-df["low"].iloc[-1])
    sigma = float(_atr) * 0.6
    paths=[]
    for _ in range(n):
        steps = np.random.normal(loc=0.0, scale=sigma, size=h)
        y = np.r_[c0, c0 + np.cumsum(steps)]
        if drift_line is not None:
            y = 0.7*y + 0.3*drift_line
        paths.append(y)
    return paths

# === ゴースト投影（疑似チャート） ===
if enable_ghost:
    try:
        pat = _pick_best_pattern(patterns)
        c0 = float(df["close"].iloc[-1])
        tgt_up_px = c0; tgt_dn_px = c0
        P_up = P_dn = np.nan

        if pat is not None:
            meas = measured_targets(pat)
            tgt_up_px = c0 + float(meas.get("up", 0.0))
            tgt_dn_px = c0 + float(meas.get("down", 0.0))

            model, Xcols, meta = load_break_model(prob_model_path)
            if model is not None:
                upper_level, lower_level = _pattern_levels_for_prob(df, pat)
                ts_now = df.index[-1]
                if upper_level is not None:
                    f_up = make_features_for_level(df, ts_now, upper_level, +1, touch_buffer)
                    P_up = float(model.predict_proba([f_up])[0,1])
                if lower_level is not None:
                    f_dn = make_features_for_level(df, ts_now, lower_level, -1, touch_buffer)
                    P_dn = float(model.predict_proba([f_dn])[0,1])
            else:
                if pat.direction_bias == "up":   P_up, P_dn = 0.65, 0.35
                elif pat.direction_bias == "down": P_up, P_dn = 0.35, 0.65
                else: P_up, P_dn = 0.5, 0.5

        idx0 = df.index[-1]
        inferred = pd.infer_freq(df.index)
        freq = inferred if inferred else "T"
        future_idx = pd.date_range(idx0, periods=ghost_h+1, freq=freq, tz=idx0.tz)

        if ghost_mode == "EV直線":
            xs, y = _ghost_path_ev(df, P_up, P_dn, tgt_up_px, tgt_dn_px, ghost_h)
            fig.add_trace(go.Scatter(
                x=future_idx, y=y, mode="lines",
                line=dict(color="rgba(255,255,255,0.9)", width=2, dash="dot"),
                name="ghost(EV)", showlegend=False, opacity=ghost_alpha
            ))

        else:  # ボラ扇形(平均)
            xs, mid, up, dn = _ghost_cone_from_atr(df, ghost_h, ghost_fan_k)
            xs_ev, y_ev = _ghost_path_ev(df, P_up, P_dn, tgt_up_px, tgt_dn_px, ghost_h)
            fig.add_trace(go.Scatter(x=future_idx, y=y_ev, mode="lines",
                                     line=dict(color="rgba(255,255,255,0.95)", width=2, dash="dot"),
                                     name="ghost(drift)", showlegend=False, opacity=ghost_alpha))
            x_poly = list(future_idx) + list(future_idx[::-1])
            y_poly = list(up) + list(dn[::-1])
            fig.add_trace(go.Scatter(
                x=x_poly, y=y_poly, fill="toself",
                line=dict(width=0), fillcolor=f"rgba(255,255,255,{ghost_alpha*0.35:.3f})",
                name="ghost(cone)", showlegend=False
            ))
            if ghost_sims and ghost_sims > 0:
                walks = _rand_walks(df, ghost_h, ghost_sims, y_ev)
                for w in walks:
                    fig.add_trace(go.Scatter(
                        x=future_idx, y=w, mode="lines",
                        line=dict(width=1, color="rgba(200,200,200,0.55)"),
                        showlegend=False, opacity=ghost_alpha*0.6
                    ))
    except Exception as e:
        st.warning(f"ゴースト投影で問題が発生しました: {e}")

# === チャートレイアウト & 表示 ===
fig.update_layout(template="plotly_dark", paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
                  title=f"{symbol} {interval} - Auto Lines (Dark)", xaxis_rangeslider_visible=False,
                  font=dict(color=COLOR_TEXT, size=12))
fig.update_xaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, showline=True, linecolor=COLOR_GRID)
fig.update_yaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, showline=True, linecolor=COLOR_GRID)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"最終更新: {pd.Timestamp.now(tz=JST).strftime('%Y-%m-%d %H:%M:%S %Z')}")

# ---------------- 近傍ニュース判定（モード別） ----------------
def near_news(ts: pd.Timestamp) -> bool:
    if news_filter_mode == "重要度別（赤影と同じ）":
        return is_suppressed(ts, windows_df)
    else:
        if news_df.empty: return False
        win = pd.Timedelta(minutes=news_win)
        cond = (news_df["importance"] >= news_imp_min) & (news_df["time"].between(ts-win, ts+win))
        return bool(cond.any())

# ---------------- シグナル（直近バー） ----------------
st.subheader("📣 シグナル（直近バー）")
i_last = len(df)-1
c_last = float(df["close"].iloc[i_last]); h_last = float(df["high"].iloc[i_last]); l_last = float(df["low"].iloc[i_last])
ts_last = df.index[i_last]

# ソフト抑制：窓内ならブレイクバッファ/K を強化
if use_soft_suppress and near_news(ts_last):
    break_buffer = break_buffer_base + soft_break_add
    retest_wait_k = retest_wait_k_base + soft_K_add
else:
    break_buffer = break_buffer_base
    retest_wait_k = retest_wait_k_base

alerts = []
if signal_mode == "水平線ブレイク(終値)":
    for lv in levels:
        if (c_last > lv + break_buffer) and (l_last <= lv): alerts.append(("上抜け", f"Lv {lv:.3f}"))
        if (c_last < lv - break_buffer) and (h_last >= lv): alerts.append(("下抜け", f"Lv {lv:.3f}"))
elif signal_mode == "トレンドラインブレイク(終値)":
    if trend:
        tl_val = trend["y1"]
        if (c_last > tl_val + break_buffer) and (l_last <= tl_val): alerts.append(("トレンド上抜け", f"TL {tl_val:.3f}"))
        if (c_last < tl_val - break_buffer) and (h_last >= tl_val): alerts.append(("トレンド下抜け", f"TL {tl_val:.3f}"))
elif signal_mode == "チャネル上抜け/下抜け(終値)":
    if trend and trend["sigma"] > 0:
        up = trend["y1"] + chan_k*trend["sigma"]
        dn = trend["y1"] - chan_k*trend["sigma"]
        if c_last > up + break_buffer: alerts.append(("チャネル上抜け", f"UP {up:.3f}"))
        if c_last < dn - break_buffer: alerts.append(("チャネル下抜け", f"DN {dn:.3f}"))
elif signal_mode == "リテスト指値(水平線)":
    if i_last >= 1:
        c_prev = float(df["close"].iloc[i_last-1]); l_prev = float(df["low"].iloc[i_last-1]); h_prev = float(df["high"].iloc[i_last-1])
        for lv in levels:
            up_break_prev = (c_prev > lv + break_buffer) and (l_prev <= lv)
            dn_break_prev = (c_prev < lv - break_buffer) and (h_prev >= lv)
            if up_break_prev and abs(c_last - lv) <= touch_buffer: alerts.append(("リテスト買い候補", f"Lv {lv:.3f}"))
            if dn_break_prev and abs(c_last - lv) <= touch_buffer: alerts.append(("リテスト売り候補", f"Lv {lv:.3f}"))

if not alerts:
    st.info("直近バーではシグナルなし。")
else:
    for kind, msg in alerts:
        if apply_news_filter and near_news(ts_last):
            st.warning(f"抑制（ニュース近傍）: {kind} - {msg} @ {ts_last}")
        else:
            st.success(f"{kind}: {msg} @ {ts_last}")

# ---------------- バックテスト（Retest指数つき / ハード抑制は従来通り） ----------------
def compute_retest(series_close: pd.Series, target: float, start_idx: int, K: int, tol_abs: float):
    hits = 0; checks = 0; hit_once = False
    last = len(series_close) - 1
    for j in range(start_idx+1, min(start_idx+K+1, last+1)):
        px = float(series_close.iloc[j])
        checks += 1
        if abs(px - target) <= tol_abs:
            hits += 1
            hit_once = True
    idx_val = (hits / checks) if checks > 0 else 0.0
    return idx_val, hit_once

def backtest(df: pd.DataFrame, levels: list, fwd_n: int, break_buffer_arg: float,
             spread_pips: float, news_df: pd.DataFrame, news_win: int,
             news_imp_min: int, apply_news: bool, signal_mode: str, retest_wait_k_arg: int,
             touch_buffer: float):
    rows=[]
    if len(df) <= fwd_n+1:
        return pd.DataFrame(columns=["time","mode","level_or_val","dir","entry","exit","ret_pips","retest_index","retest_hit"])
    pv = pip_value("USDJPY")
    close_s = df["close"]
    use_imp_mode = (news_filter_mode == "重要度別（赤影と同じ）")
    for i in range(1, len(df)-fwd_n):
        t = df.index[i]
        c  = float(df["close"].iloc[i])
        l1 = float(df["low"].iloc[i-1]); h1 = float(df["high"].iloc[i-1])
        # ハード抑制
        if apply_news:
            if use_imp_mode:
                if is_suppressed(t, windows_df): 
                    continue
            else:
                if not news_df.empty:
                    win = pd.Timedelta(minutes=news_win)
                    if ((news_df["importance"] >= news_imp_min) & (news_df["time"].between(t-win, t+win))).any():
                        continue
        # 以降は従来通り
        if signal_mode == "水平線ブレイク(終値)":
            for lv in levels:
                if (c > lv + break_buffer_arg) and (l1 <= lv):
                    entry, exitp = c, float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, lv, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="水平ブレイク上", level_or_val=float(lv),
                                     dir="long", entry=entry, exit=exitp,
                                     ret_pips=(exitp-entry)/pv - spread_pips,
                                     retest_index=ri, retest_hit=rh))
                if (c < lv - break_buffer_arg) and (h1 >= lv):
                    entry, exitp = c, float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, lv, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="水平ブレイク下", level_or_val=float(lv),
                                     dir="short", entry=entry, exit=exitp,
                                     ret_pips=(entry-exitp)/pv - spread_pips,
                                     retest_index=ri, retest_hit=rh))
        elif signal_mode == "トレンドラインブレイク(終値)":
            if trend:
                tl = trend["y1"]
                if (c > tl + break_buffer_arg) and (l1 <= tl):
                    entry, exitp = c, float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, tl, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="TLブレイク上", level_or_val=float(tl),
                                     dir="long", entry=entry, exit=exitp,
                                     ret_pips=(exitp-entry)/pv - spread_pips,
                                     retest_index=ri, retest_hit=rh))
                if (c < tl - break_buffer_arg) and (h1 >= tl):
                    entry, exitp = c, float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, tl, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="TLブレイク下", level_or_val=float(tl),
                                     dir="short", entry=entry, exit=exitp,
                                     ret_pips=(entry-exitp)/pv - spread_pips,
                                     retest_index=ri, retest_hit=rh))
        elif signal_mode == "チャネル上抜け/下抜け(終値)":
            if trend and trend["sigma"] > 0:
                up = trend["y1"] + chan_k*trend["sigma"]
                dn = trend["y1"] - chan_k*trend["sigma"]
                if c > up + break_buffer_arg:
                    entry, exitp = c, float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, up, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="チャネル上抜け", level_or_val=float(up),
                                     dir="long", entry=entry, exit=exitp,
                                     ret_pips=(exitp-entry)/pv - spread_pips,
                                     retest_index=ri, retest_hit=rh))
                if c < dn - break_buffer_arg:
                    entry, exitp = c, float(df["close"].iloc[i+fwd_n])
                    ri, rh = compute_retest(close_s, dn, i, int(retest_wait_k_arg), float(touch_buffer))
                    rows.append(dict(time=t, mode="チャネル下抜け", level_or_val=float(dn),
                                     dir="short", entry=entry, exit=exitp,
                                     ret_pips=(entry-exitp)/pv - spread_pips,
                                     retest_index=ri, retest_hit=rh))
        elif signal_mode == "リテスト指値(水平線)":
            K = int(retest_wait_k_arg)
            for lv in levels:
                up_break = (c > lv + break_buffer_arg) and (l1 <= lv)
                dn_break = (c < lv - break_buffer_arg) and (h1 >= lv)
                if up_break:
                    for j in range(i+1, min(i+K, len(df)-fwd_n)):
                        if abs(float(df["close"].iloc[j]) - lv) <= touch_buffer:
                            entry = float(df["close"].iloc[j]); exitp = float(df["close"].iloc[j+fwd_n])
                            ri, rh = compute_retest(close_s, lv, i, K, float(touch_buffer))
                            rows.append(dict(time=df.index[j], mode="リテスト(L)", level_or_val=float(lv),
                                             dir="long", entry=entry, exit=exitp,
                                             ret_pips=(exitp-entry)/pv - spread_pips,
                                             retest_index=ri, retest_hit=rh))
                            break
                if dn_break:
                    for j in range(i+1, min(i+K, len(df)-fwd_n)):
                        if abs(float(df["close"].iloc[j]) - lv) <= touch_buffer:
                            entry = float(df["close"].iloc[j]); exitp = float(df["close"].iloc[j+fwd_n])
                            ri, rh = compute_retest(close_s, lv, i, K, float(touch_buffer))
                            rows.append(dict(time=df.index[j], mode="リテスト(S)", level_or_val=float(lv),
                                             dir="short", entry=entry, exit=exitp,
                                             ret_pips=(entry-exitp)/pv - spread_pips,
                                             retest_index=ri, retest_hit=rh))
                            break
    return pd.DataFrame(rows)

bt_df = None
if run_bt:
    with st.spinner("バックテスト実行中..."):
        bt_df = backtest(df, levels, fwd_n, break_buffer, spread_pips,
                         news_df, news_win, news_imp_min, apply_news_filter,
                         signal_mode, retest_wait_k, touch_buffer)
    if bt_df is None or bt_df.empty:
        st.warning("トレードが生成されませんでした。パラメータ/モードを見直してください。")
    else:
        st.subheader("📊 バックテスト結果（" + signal_mode + "）")
        total = len(bt_df)
        wins = int((bt_df["ret_pips"] > 0).sum())
        mean = float(bt_df["ret_pips"].mean())
        std  = float(bt_df["ret_pips"].std()) if total > 1 else float("nan")
        sharpe = mean / std if std and not math.isnan(std) and std>1e-9 else float("nan")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("件数", total)
        c2.metric("勝率", f"{wins/total*100:.1f}%")
        c3.metric("平均pips", f"{mean:.2f}")
        c4.metric("Sharpe(擬似)", f"{sharpe:.2f}" if not math.isnan(sharpe) else "n/a")

        st.write("方向別：")
        st.dataframe(bt_df.groupby("dir")["ret_pips"].agg(["count","mean","std"]))

        st.write("シグナル別：")
        st.dataframe(bt_df.groupby("mode")["ret_pips"].agg(["count","mean","std"]).sort_values("mean", ascending=False))

        st.write("リテストあり/なし 比較：")
        comp = bt_df.copy()
        comp["retest_bucket"] = comp["retest_hit"].map({True:"あり", False:"なし"})
        st.dataframe(comp.groupby("retest_bucket")["ret_pips"].agg(
            件数="count", 勝率=lambda s: (s>0).mean()*100, 平均pips="mean", STD="std", Retest指数平均=lambda s: comp.loc[s.index,"retest_index"].mean()
        ).round({"勝率":1, "平均pips":2, "STD":2, "Retest指数平均":2}))

        ch = bt_df[bt_df["mode"].isin(["チャネル上抜け","チャネル下抜け"])]
        if not ch.empty:
            st.write("チャネル上抜け/下抜けの統計：")
            st.dataframe(ch.groupby("mode")["ret_pips"].agg(["count","mean","std"]).rename(
                index={"チャネル上抜け":"上抜け","チャネル下抜け":"下抜け"}).sort_values("mean", ascending=False))

# ---------------- 発注（任意・簡易） ----------------
st.subheader("🧪 発注（任意）")
side = st.selectbox("方向", ["BUY","SELL"], index=0)
order_units = st.sidebar.number_input("発注数量", value=1000, step=1000)
paper_trade = st.sidebar.checkbox("紙トレ（シミュレーション）を有効", value=True)
use_oanda = st.sidebar.checkbox("OANDA発注を有効", value=False)
oanda_token = st.sidebar.text_input("OANDA Token（env: OANDA_TOKEN）", value=os.getenv("OANDA_TOKEN",""))
oanda_account = st.sidebar.text_input("OANDA AccountID（env: OANDA_ACCOUNT）", value=os.getenv("OANDA_ACCOUNT",""))
oanda_env = st.sidebar.selectbox("OANDA環境", ["practice","live"], index=0)
use_mt5 = st.sidebar.checkbox("MT5発注を有効（Windows）", value=False)
mt5_symbol = st.sidebar.text_input("MT5シンボル", value="USDJPY")

if st.button("成行発注"):
    price_now = float(df["close"].iloc[-1]); logs = []
    if paper_trade:
        st.success(f"[紙トレ] {side} {order_units} @ {price_now:.3f}")
        logs.append({"type":"paper","side":side,"units":order_units,"price":price_now,"time":str(df.index[-1])})
    if use_oanda and oanda_token and oanda_account:
        try:
            import requests
            inst = "USD_JPY"
            base = "https://api-fxpractice.oanda.com" if oanda_env=="practice" else "https://api-fxtrade.oanda.com"
            headers = {"Authorization": f"Bearer {oanda_token}", "Content-Type":"application/json"}
            data = {"order":{"units": str(order_units if side=="BUY" else -order_units),
                             "instrument": inst, "timeInForce":"FOK","type":"MARKET","positionFill":"DEFAULT"}}
            r = requests.post(f"{base}/v3/accounts/{oanda_account}/orders", headers=headers, data=json.dumps(data), timeout=10)
            st.success(f"[OANDA] 成行 {side} 送信OK") if r.status_code in (200,201) else st.error(f"[OANDA] {r.status_code} {r.text}")
            logs.append({"type":"oanda","resp":r.json() if r.ok else r.text})
        except Exception as e:
            st.error(f"[OANDA] 例外: {e}")
    if use_mt5:
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                st.error(f"[MT5] 初期化失敗: {mt5.last_error()}")
            else:
                symbol_mt5 = mt5_symbol
                mt5.symbol_select(symbol_mt5, True)
                lot = 0.1
                typ = mt5.ORDER_TYPE_BUY if side=="BUY" else mt5.ORDER_TYPE_SELL
                req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol_mt5, "volume": lot, "type": typ,
                       "deviation": 10, "magic": 123456, "comment": "AutoLines", "type_filling": mt5.ORDER_FILLING_FOK}
                res = mt5.order_send(req)
                st.success(f"[MT5] 成行 {side} OK: {res.order}") if res.retcode == mt5.TRADE_RETCODE_DONE else st.error(f"[MT5] 失敗: {res.retcode}")
                mt5.shutdown()
        except Exception as e:
            st.error(f"[MT5] 例外: {e}")
    st.session_state.setdefault("trade_logs", [])
    st.session_state["trade_logs"].extend(logs)

with st.expander("実行ログ"):
    st.write(st.session_state.get("trade_logs", []))

# ---------------- スコア表 ----------------
st.subheader("⭐ 重要度スコア（上位）")
if not score_df.empty:
    st.dataframe(score_df[["level","score","touches","session_ratio"]].head(15))
else:
    st.info("レベルが検出されていません。eps/min_samples/look を調整してください。")

# ---------------- 期待pips（水平線ブレイクの過去平均） ----------------
def compute_expected_pips_table_for_levels(df, levels, fwd_n, break_buffer, spread_pips,
                                           news_df, news_win, news_imp_min, apply_news_filter,
                                           touch_buffer, retest_wait_k):
    bt = backtest(df, levels, fwd_n, break_buffer, spread_pips,
                  news_df, news_win, news_imp_min, apply_news_filter,
                  "水平線ブレイク(終値)", retest_wait_k, touch_buffer)
    if bt is None or bt.empty:
        return None, None
    by_level_dir = (bt.groupby(["level_or_val","dir"])["ret_pips"]
                      .agg(avg="mean", n="count").reset_index()
                      .rename(columns={"level_or_val":"level"}))
    by_dir = (bt.groupby("dir")["ret_pips"].agg(avg="mean", n="count").reset_index())
    return by_level_dir, by_dir

# ---------------- 「今からのブレイク確率」＋ 期待値ランキング ----------------
if show_break_prob:
    st.subheader("🎯 今からの水平線ブレイク確率（次H本以内）")
    model, Xcols, meta = load_break_model(prob_model_path)
    if model is None:
        st.info("学習モデル（models/break_model.joblib）が見つかりません。先に ai_train_break.py を実行してください。")
    else:
        ts_now = df.index[-1]
        use_levels = list(score_df["level"].head(break_prob_topk)) if (score_df is not None and not score_df.empty) else levels[:break_prob_topk]
        rows=[]
        for lv in use_levels:
            feat_up = make_features_for_level(df, ts_now, lv, +1, touch_buffer)
            feat_dn = make_features_for_level(df, ts_now, lv, -1, touch_buffer)
            p_up = float(model.predict_proba([feat_up])[0,1])
            p_dn = float(model.predict_proba([feat_dn])[0,1])
            rows.append(dict(level=float(lv), P_up=p_up, P_dn=p_dn))
        prob_df = pd.DataFrame(rows)

        ev_table, ev_dir = compute_expected_pips_table_for_levels(
            df, levels, fwd_n, break_buffer, spread_pips,
            news_df, news_win, news_imp_min, apply_news_filter,
            touch_buffer, retest_wait_k
        )

        def get_expected_for(level, direction):
            if ev_table is not None and not ev_table.empty:
                sub = ev_table[(ev_table["level"]==float(level)) & (ev_table["dir"]==direction)]
                if not sub.empty and int(sub["n"].iloc[0]) >= ev_level_min_samples:
                    return float(sub["avg"].iloc[0]), int(sub["n"].iloc[0])
            if ev_dir is not None and not ev_dir.empty:
                subd = ev_dir[ev_dir["dir"]==direction]
                if not subd.empty:
                    return float(subd["avg"].iloc[0]), int(subd["n"].iloc[0])
            return 0.0, 0

        exp_rows=[]
        for _, r in prob_df.iterrows():
            lv = float(r["level"])
            e_up, n_up = get_expected_for(lv, "long")
            e_dn, n_dn = get_expected_for(lv, "short")
            ev_up = r["P_up"] * e_up
            ev_dn = r["P_dn"] * e_dn
            best_dir = "BUY" if ev_up >= ev_dn else "SELL"
            best_ev  = ev_up if ev_up >= ev_dn else ev_dn
            exp_rows.append({
                "level": lv,
                "P_up": r["P_up"], "P_dn": r["P_dn"],
                "E_pips_up": e_up, "E_pips_dn": e_dn,
                "EV_up": ev_up, "EV_dn": ev_dn,
                "best_action": best_dir, "best_EV": best_ev,
                "samples_up": n_up, "samples_dn": n_dn
            })
        ev_df = (pd.DataFrame(exp_rows)
                   .sort_values(["best_EV","EV_up","EV_dn"], ascending=False)
                   .reset_index(drop=True))

        st.dataframe(
            ev_df[["level","P_up","P_dn","E_pips_up","E_pips_dn","EV_up","EV_dn","best_action","best_EV","samples_up","samples_dn"]]
                .style.format({
                    "level":"{:.3f}",
                    "P_up":"{:.2%}","P_dn":"{:.2%}",
                    "E_pips_up":"{:.2f}","E_pips_dn":"{:.2f}",
                    "EV_up":"{:.2f}","EV_dn":"{:.2f}",
                    "best_EV":"{:.2f}"
                })
        )
        st.caption(
            f"H（学習）: {meta.get('H','?')} / H（表示）: {break_prob_h} / "
            f"break_buffer(学習): {meta.get('break_buffer','?')} / touch_buffer(学習): {meta.get('touch_buffer','?')}  |  "
            f"※ 期待pipsは『水平線ブレイク(終値)』のバックテスト平均から推定"
        )

if show_ev_rank and not show_break_prob:
    st.info("期待値ランキングは「今からの水平線ブレイク確率」をONにすると表示されます。")

# ---------------- パターンの「次の動きの予測」一覧（測定値ターゲット & EV） ----------------
st.subheader("🧭 パターン検出：次の動きの予測（測定値ターゲット & EV）")
if not patterns:
    st.info("現在、検出されたパターンはありません。lookback/感度を調整してください。")
else:
    ev_table, ev_dir = compute_expected_pips_table_for_levels(
        df, levels, fwd_n, break_buffer, spread_pips,
        news_df, news_win, news_imp_min, apply_news_filter,
        touch_buffer, retest_wait_k
    )
    model, Xcols, meta = load_break_model(prob_model_path)

    rows=[]
    ts_now = df.index[-1]
    for p in patterns:
        meas = measured_targets(p)
        tgt_up_price = df["close"].iloc[-1] + meas["up"]
        tgt_dn_price = df["close"].iloc[-1] + meas["down"]

        upper_level = None; lower_level = None
        if p.kind.startswith("triangle"):
            subp = df.loc[p.params["sub_start"]:p.params["sub_end"]]
            m1,b1 = p.params["upper"]; m2,b2 = p.params["lower"]
            x_now  = len(subp)-1
            y_u = float(m1*x_now + b1); y_l = float(m2*x_now + b2)
            upper_level, lower_level = y_u, y_l
        elif p.kind=="rectangle":
            upper_level, lower_level = float(p.params["upper"]), float(p.params["lower"])
        elif p.kind=="double_top":
            upper_level, lower_level = None, float(p.params["neck"])
        elif p.kind=="double_bottom":
            upper_level, lower_level = float(p.params["neck"]), None
        elif p.kind in ("flag_up","flag_dn","pennant"):
            upper_level, lower_level = float(p.params["upper_now"]), float(p.params["lower_now"])
        elif p.kind=="head_shoulders":
            upper_level, lower_level = None, float(p.params["neck"])
        elif p.kind=="inverse_head_shoulders":
            upper_level, lower_level = float(p.params["neck"]), None

        P_up = P_dn = np.nan
        if model is not None:
            if upper_level is not None:
                f_up = make_features_for_level(df, ts_now, upper_level, +1, touch_buffer)
                P_up = float(model.predict_proba([f_up])[0,1])
            if lower_level is not None:
                f_dn = make_features_for_level(df, ts_now, lower_level, -1, touch_buffer)
                P_dn = float(model.predict_proba([f_dn])[0,1])

        def _ev_for(level, direction):
            if level is None: return (0.0, 0, np.nan)
            if ev_table is not None and not ev_table.empty:
                sublv = ev_table[(ev_table["dir"]==direction)]
                if not sublv.empty:
                    idx_min = (sublv["level"]-float(level)).abs().idxmin()
                    avg, n = float(sublv.loc[idx_min,"avg"]), int(sublv.loc[idx_min,"n"])
                    if n >= ev_level_min_samples:
                        p = P_up if direction=="long" else P_dn
                        return (avg, n, (p*avg) if not np.isnan(p) else np.nan)
            if ev_dir is not None and not ev_dir.empty:
                dsub = ev_dir[ev_dir["dir"]==direction]
                if not dsub.empty:
                    avg, n = float(dsub["avg"].iloc[0]), int(dsub["n"].iloc[0])
                    p = P_up if direction=="long" else P_dn
                    return (avg, n, (p*avg) if not np.isnan(p) else np.nan)
            return (0.0, 0, np.nan)

        E_up, N_up, EV_up = _ev_for(upper_level, "long")
        E_dn, N_dn, EV_dn = _ev_for(lower_level, "short")

        rows.append(dict(
            pattern=p.kind, quality=round(p.quality,1),
            upper_level=upper_level, lower_level=lower_level,
            P_up=P_up, P_dn=P_dn,
            E_pips_up=E_up, E_pips_dn=E_dn,
            EV_up=EV_up, EV_dn=EV_dn,
            target_up=tgt_up_price, target_dn=tgt_dn_price
        ))

    out = pd.DataFrame(rows)
    if not out.empty:
        out["best_action"] = out.apply(lambda r: "BUY" if (pd.notna(r["EV_up"]) and (r["EV_up"]>= (r["EV_dn"] if pd.notna(r["EV_dn"]) else -1e9))) else "SELL", axis=1)
        out["best_EV"] = out.apply(lambda r: r["EV_up"] if r["best_action"]=="BUY" else r["EV_dn"], axis=1)
        st.dataframe(
            out[["pattern","quality","upper_level","lower_level","P_up","P_dn",
                 "E_pips_up","E_pips_dn","EV_up","EV_dn","best_action","best_EV",
                 "target_up","target_dn"]]
            .style.format({
                "upper_level":"{:.3f}", "lower_level":"{:.3f}",
                "P_up":"{:.1%}", "P_dn":"{:.1%}",
                "E_pips_up":"{:.2f}", "E_pips_dn":"{:.2f}",
                "EV_up":"{:.2f}", "EV_dn":"{:.2f}", "best_EV":"{:.2f}",
                "target_up":"{:.3f}", "target_dn":"{:.3f}",
            })
        )
        st.caption("※ P_up/P_dn は学習モデル（水平ブレイク）を、パターンの境界（上辺/下辺/ネック/チャネル）に当てはめて推定。ターゲットは測定値（旗竿・厚み・ヘッド高さ 等）。")
    else:
        st.info("パターンは検出されましたが、テーブル化できる十分な指標がありません。")

import streamlit as st

st.markdown("---")
st.markdown("""
## 免責事項 / Disclaimer

本アプリは、過去の価格データ等をもとに作成された分析ツール・教育用コンテンツであり、
金融商品取引法に基づく投資助言・代理業務を行うものではありません。

本アプリが提供する情報・シグナル・予測結果は、将来の成果や利益を保証するものではなく、
投資判断はすべて利用者ご自身の責任において行ってください。

当方は、本アプリの利用により生じたいかなる損失・損害についても、一切の責任を負いません。

また、本アプリが使用するデータは Yahoo Finance 等の外部サービスを通じて取得しています。
データの正確性・完全性は保証されませんのでご了承ください。

本アプリの利用により生じるあらゆるリスクは、利用者ご自身の自己責任においてご対応ください。
""")