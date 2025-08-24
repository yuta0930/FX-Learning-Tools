# ai_train.py  (intraday期間を自動で調整する版)
import os, math, joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# ====== ベース設定（必要に応じて変更） ======
SYMBOL   = "JPY=X"     # USDJPY
PERIOD   = "180d"      # 希望期間（intradayなら自動で短縮します）
INTERVAL = "15m"       # 15mはYahoo制約で最大60dまで
LOOK     = 5
EPS      = 0.08
MIN_SAMP = 4
TOUCH_BUFFER = 0.05
BREAK_BUFFER = 0.05
FWD_N    = 20
SPREAD_PIPS = 0.5
PIP = 0.01

os.makedirs("models", exist_ok=True)

# --- ユーティリティ ---
INTRADAY_SET = {"1m","2m","5m","15m","30m","60m","90m"}
# Yahooのだいたいの上限（実際は多少前後することがあります）
MAX_PERIOD_BY_INTERVAL = {
    "1m": "7d",   # 実質7日
    "2m": "60d",
    "5m": "60d",
    "15m":"60d",
    "30m":"60d",
    "60m":"730d",  # 60分は約2年
    "90m":"730d",
}

def clamp_period_for_interval(period: str, interval: str) -> str:
    """インターバルに応じて取得可能な最大期間へクランプ"""
    if interval not in INTRADAY_SET:
        return period
    maxp = MAX_PERIOD_BY_INTERVAL.get(interval)
    if maxp is None:
        return period
    # 期間比較はシンプルに日単位で比較（"Xd" 形式のみ想定）
    def to_days(p: str) -> int:
        p = p.strip().lower()
        if p.endswith("d"): return int(p[:-1])
        if p.endswith("mo"): return int(p[:-2]) * 30
        if p.endswith("y"): return int(p[:-1]) * 365
        # yfinanceのperiod表記に合わない場合は大きめに扱う
        return 999999
    return maxp if to_days(period) > to_days(maxp) else period

def atr(df, period=14):
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """history→空ならdownloadでフォールバック。intradayは期間を自動短縮。"""
    adj_period = clamp_period_for_interval(period, interval)
    if adj_period != period:
        print(f"[info] {interval} のため期間を {period} → {adj_period} に自動調整します。")
    # 1) history
    try:
        df = yf.Ticker(symbol).history(period=adj_period, interval=interval, auto_adjust=False)
    except Exception as e:
        print(f"[warn] historyで例外: {e}")
        df = pd.DataFrame()
    # 2) download フォールバック
    if df is None or df.empty:
        try:
            df = yf.download(symbol, period=adj_period, interval=interval, auto_adjust=False, progress=False)
        except Exception as e:
            print(f"[warn] downloadで例外: {e}")
            df = pd.DataFrame()
    if df is None or df.empty:
        raise RuntimeError(f"yfinanceからデータ取得失敗（symbol={symbol}, interval={interval}, period={adj_period}）")
    # 列名を標準化（historyは単層のはず）
    df = df.rename(columns=lambda c: str(c).strip())
    need = {"Open","High","Low","Close"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"OHLC列が不足: {set(df.columns)}")
    return df

def pivots(df, look=5):
    highs = df["High"].rolling(look, center=True).max()
    lows  = df["Low"].rolling(look, center=True).min()
    ph = df[df["High"]==highs]
    pl = df[df["Low"]==lows]
    return ph, pl

def levels_from_pivots(ph, pl, eps=0.08, min_samples=4):
    prices = np.r_[ph["High"].values, pl["Low"].values].reshape(-1,1)
    if len(prices)==0: return []
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(prices).labels_
    out=[]
    for lab in set(labels)-{-1}:
        out.append(float(prices[labels==lab].mean()))
    return sorted(out)

def session_name(ts):
    h = ts.hour
    if 9 <= h < 15: return "Tokyo"
    if 16<= h < 24: return "London"
    if h>=22 or h<5: return "NY"
    return "Other"

def engineer_row(df, i, lv, atr_recent):
    ts = df.index[i]
    c  = float(df["Close"].iloc[i])
    h  = float(df["High"].iloc[i])
    l  = float(df["Low"].iloc[i])
    dist     = abs(c - lv)
    near     = 1.0 / (dist + 1e-6)
    Ntouch=200
    sub = df.iloc[max(0, i-Ntouch):i+1]
    touches = int((((sub["Low"]<=lv)&(sub["High"]>=lv)) | (sub["Close"].sub(lv).abs()<=TOUCH_BUFFER)).sum())
    sess = session_name(ts)
    sess_tokyo = 1.0 if sess=="Tokyo" else 0.0
    sess_london= 1.0 if sess=="London" else 0.0
    sess_ny    = 1.0 if sess=="NY" else 0.0
    atr_norm = atr_recent / max(1e-6, c)
    return dict(
        ts=ts, level=lv, close=c, high=h, low=l,
        touches=touches, near=near, atr_norm=atr_norm,
        sess_tokyo=sess_tokyo, sess_london=sess_london, sess_ny=sess_ny
    )

def make_dataset(df, levels, fwd_n=20):
    _atr = atr(df, 14)
    rows=[]
    for i in range(1, len(df)-fwd_n):
        c  = float(df["Close"].iloc[i])
        l1 = float(df["Low"].iloc[i-1])
        h1 = float(df["High"].iloc[i-1])
        atr_recent = float(_atr.iloc[i]) if not _atr.isna().iloc[i] else 0.0
        exitp = float(df["Close"].iloc[i+fwd_n])
        for lv in levels:
            base = engineer_row(df, i, lv, atr_recent)
            # 上ブレイク
            up_break = (c > lv + BREAK_BUFFER) and (l1 <= lv)
            if up_break:
                ret_pips = (exitp - c)/PIP - SPREAD_PIPS
                y = 1 if ret_pips > 0 else 0
                rows.append({**base, "dir":1, "y":y})
            # 下ブレイク
            dn_break = (c < lv - BREAK_BUFFER) and (h1 >= lv)
            if dn_break:
                ret_pips = (c - exitp)/PIP - SPREAD_PIPS
                y = 1 if ret_pips > 0 else 0
                rows.append({**base, "dir":-1, "y":y})
    return pd.DataFrame(rows)

def main():
    df = fetch_data(SYMBOL, PERIOD, INTERVAL)
    ph, pl = pivots(df, LOOK)
    lv = levels_from_pivots(ph, pl, EPS, MIN_SAMP)
    if not lv:
        raise RuntimeError("レベル抽出ゼロ。EPS/MIN_SAMP/LOOKを見直してください。")
    ds = make_dataset(df, lv, FWD_N)
    if ds.empty:
        raise RuntimeError("データセットが空。期間/足/バッファ/Nを見直してください。")

    Xcols = ["touches","near","atr_norm","sess_tokyo","sess_london","sess_ny","dir"]
    X = ds[Xcols].values
    y = ds["y"].values

    tscv = TimeSeriesSplit(n_splits=5)
    aucs=[]
    for tr, va in tscv.split(X):
        model = LogisticRegression(max_iter=200, solver="lbfgs")
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[va])[:,1]
        aucs.append(roc_auc_score(y[va], p))
    print(f"CV AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X, y)
    joblib.dump({"model":model, "Xcols":Xcols, "meta":{"symbol":SYMBOL,"interval":INTERVAL}}, "models/line_model.joblib")
    print("saved -> models/line_model.joblib")
    print(f"samples={len(ds)}, positives={int(ds['y'].sum())}")

if __name__ == "__main__":
    main()
