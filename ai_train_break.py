# ai_train_break.py  --- チューニング版（未来情報抑制 + 特徴量拡充 + RF + 確率較正）
import os, math, joblib, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import DBSCAN
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# ====== 設定 ======
SYMBOL = "JPY=X"      # USDJPY
INTERVAL = "15m"      # 15mは最大60d
PERIOD = "60d"
LOOK = 5              # ピボット窓
EPS = 0.08            # 水平クラスタの許容幅（価格）
MIN_SAMP = 4
H = 12                # 先読み本数（この本数以内にブレイクしたか）
BREAK_BUFFER = 0.05   # ブレイク判定バッファ（価格）
TOUCH_BUFFER = 0.05   # 接触カウント用バッファ
TREND_LOOK = 150      # 回帰傾き/チャネル用の窓
LEVEL_WIN = 1000      # ★ 過去だけでレベル抽出するウィンドウ幅
LEVEL_STRIDE = 2      # ★ 何本ごとにレベルを更新するか（高速化）
STEP = 1              # 学習サンプルの間引き（1なら全バー）
SEED = 42
os.makedirs("models", exist_ok=True)

# yfinance制約
INTRADAY_SET = {"1m","2m","5m","15m","30m","60m","90m"}
MAX_PERIOD_BY_INTERVAL = {
    "1m":"7d","2m":"60d","5m":"60d","15m":"60d","30m":"60d",
    "60m":"730d","90m":"730d",
}
def clamp_period(period, interval):
    if interval not in INTRADAY_SET: return period
    maxp = MAX_PERIOD_BY_INTERVAL.get(interval, period)
    def days(p):
        p=p.lower().strip()
        if p.endswith("d"): return int(p[:-1])
        if p.endswith("mo"): return int(p[:-2])*30
        if p.endswith("y"): return int(p[:-1])*365
        return 999999
    return maxp if days(period) > days(maxp) else period

def fetch_ohlc(sym, period, interval):
    adj = clamp_period(period, interval)
    df = yf.Ticker(sym).history(period=adj, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        df = yf.download(sym, period=adj, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance取得失敗")
    df = df.rename(columns=str).rename(columns=str.lower)
    need = {"open","high","low","close"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"OHLC不足: {set(df.columns)}")
    return df

# ====== テクニカル ======
def atr(df, n=14):
    h,l,c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / (down + 1e-9)
    return 100 - (100/(1+rs))

def stoch_k(df, n=14):
    ll = df["low"].rolling(n).min()
    hh = df["high"].rolling(n).max()
    return (df["close"]-ll) / (hh-ll + 1e-9) * 100

def regression_line(series, look):
    y = series.tail(look).values
    x = np.arange(len(y))
    if len(x) < 2: 
        return {"slope":0.0,"intercept":y[-1] if len(y)>0 else 0.0,"sigma":0.0}
    m, b = np.polyfit(x, y, 1)
    resid = y - (m*x + b)
    return {"slope":float(m), "intercept":float(b), "sigma":float(np.std(resid))}

def channel_value_at_end(df, look, k_sigma):
    """直近バーにおけるトレンドラインと上下バンド値"""
    info = regression_line(df["close"], look)
    n = min(look, len(df))
    y_t = info["slope"]*(n-1) + info["intercept"]
    up = y_t + k_sigma*info["sigma"]
    dn = y_t - k_sigma*info["sigma"]
    return y_t, up, dn, info

# ====== ピボット & レベル ======
def pivots(df, look=5):
    highs = df["high"].rolling(look, center=True).max()
    lows  = df["low"].rolling(look, center=True).min()
    ph = df[df["high"]==highs]
    pl = df[df["low"]==lows]
    return ph, pl

def levels_from_pivots(ph, pl, eps=0.08, min_samples=4):
    prices = np.r_[ph["high"].values, pl["low"].values].reshape(-1,1)
    if len(prices)==0: return []
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(prices).labels_
    out=[]
    for lab in set(labels)-{-1}:
        out.append(float(prices[labels==lab].mean()))
    return sorted(out)

# ====== 特徴量構築（i時点の“過去だけ”を使う） ======
def make_features_at_i(df, i, levels, touch_buf, trend_look):
    rows=[]
    c  = float(df["close"].iloc[i])
    ts = df.index[i]
    sub_hist = df.iloc[max(0, i-trend_look+1):i+1]

    # テクニカル群（直近）
    a = atr(df, 14).fillna(0.0).iloc[i]
    r = rsi(df["close"], 14).fillna(50.0).iloc[i]
    k = stoch_k(df, 14).fillna(50.0).iloc[i]
    # トレンド/チャネル位置
    y_t, up, dn, info = channel_value_at_end(df.iloc[:i+1], look=trend_look, k_sigma=2.0)
    ch_sigma = max(1e-9, info["sigma"])
    zpos = (c - y_t)/ch_sigma  # 回帰線からのz-score

    # 時間帯（cyclical）
    hour = ts.hour
    cyc_sin = np.sin(2*np.pi*hour/24)
    cyc_cos = np.cos(2*np.pi*hour/24)
    tokyo = 1.0 if 9<=hour<15 else 0.0
    london= 1.0 if 16<=hour<24 else 0.0
    ny    = 1.0 if hour>=22 or hour<5 else 0.0

    # 距離・接触
    for lv in levels:
        dist = c - lv
        dist_abs = abs(dist)
        near = 1.0/(dist_abs + 1e-6)
        # ATR正規化距離
        dist_atr = dist_abs / max(1e-6, a)

        Ntouch = 200
        sub = df.iloc[max(0, i-Ntouch):i+1]
        touches = int((((sub["low"]<=lv)&(sub["high"]>=lv)) | (sub["close"].sub(lv).abs()<=touch_buf)).sum())
        last_touch_bar = 0
        for kidx in range(i, max(-1, i-Ntouch), -1):
            if (df["low"].iloc[kidx] <= lv <= df["high"].iloc[kidx]) or (abs(df["close"].iloc[kidx]-lv) <= touch_buf):
                last_touch_bar = i-kidx
                break

        base = dict(
            ts=ts, level=lv, close=c, atr=a/max(1e-6,c),
            rsi=r, stoch_k=k, zpos=zpos,
            dist=dist_abs, near=near, dist_atr=dist_atr,
            touches=touches, last_touch_bar=last_touch_bar,
            cyc_sin=cyc_sin, cyc_cos=cyc_cos,
            tokyo=tokyo, london=london, ny=ny
        )
        # dir=+1/-1 として2サンプル
        rows.append(base | dict(dir=1,  sign= 1 if dist>=0 else -1))
        rows.append(base | dict(dir=-1, sign= 1 if dist<=0 else -1))
    return rows

# ====== データセット生成（未来情報を使わない） ======
def build_dataset_past_only(df, H, break_buf, touch_buf, look, level_win, level_stride, step):
    rows=[]
    last = len(df)-H-1
    cached_levels = None
    last_levels_i = -999

    for i in range(max(look, 50), last, step):
        # 過去ウィンドウでレベル再計算（ストライド制御）
        if cached_levels is None or (i - last_levels_i) >= level_stride:
            hist = df.iloc[max(0, i-level_win):i+1]
            ph, pl = pivots(hist, look)
            cached_levels = levels_from_pivots(ph, pl, EPS, MIN_SAMP)
            last_levels_i = i
        levels = cached_levels
        if not levels:
            continue

        # 特徴量（i時点）
        feats = make_features_at_i(df, i, levels, touch_buf, TREND_LOOK)
        # ラベル作成（i+1〜i+H ）
        for rec in feats:
            lv = rec["level"]
            # 上/下ブレイクの定義（終値ブレイク + 前足のレンジ内）
            up=0; dn=0
            for j in range(1, H+1):
                cc = float(df["close"].iloc[i+j])
                ll = float(df["low"].iloc[i+j-1])
                if (cc > lv + break_buf) and (ll <= lv):
                    up=1; break
            for j in range(1, H+1):
                cc = float(df["close"].iloc[i+j])
                hh = float(df["high"].iloc[i+j-1])
                if (cc < lv - break_buf) and (hh >= lv):
                    dn=1; break
            rec["y"] = up if rec["dir"]==1 else dn
            rows.append(rec)

    ds = pd.DataFrame(rows)
    return ds

# ====== メイン ======
def main():
    print(f"Fetch: {SYMBOL} {INTERVAL} {PERIOD}")
    df = fetch_ohlc(SYMBOL, PERIOD, INTERVAL)

    # データセット（未来情報抑制版）
    ds = build_dataset_past_only(df, H, BREAK_BUFFER, TOUCH_BUFFER, LOOK, LEVEL_WIN, LEVEL_STRIDE, STEP)
    if ds.empty:
        raise RuntimeError("データセットが空でした。期間/足/EPSを見直してください。")

    Xcols = [
        "dir","sign","atr","rsi","stoch_k","zpos","dist","near","dist_atr",
        "touches","last_touch_bar","cyc_sin","cyc_cos","tokyo","london","ny"
    ]
    # 欠損埋め
    for c in Xcols:
        if c not in ds.columns: ds[c]=0.0
    ds = ds.dropna(subset=["y"])

    X = ds[Xcols].values
    y = ds["y"].astype(int).values

    pos = int(y.sum()); neg = int((y==0).sum())
    print(f"dataset: n={len(ds)}, pos={pos}, neg={neg}")

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "正例/負例が片方しか無く、学習できません。\n"
            f"- Hを増やす（現在 {H}）\n- BREAK_BUFFERを小さく（現在 {BREAK_BUFFER}）\n- 期間/足を見直す（60m/180dなど）"
        )

    # ====== 時系列CVで評価 ======
    tscv = TimeSeriesSplit(n_splits=5)
    aucs=[]; pr_aucs=[]; folds=0
    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        y_tr, y_va = y[tr], y[va]
        # 片クラスならスキップ
        if len(np.unique(y_tr))<2 or len(np.unique(y_va))<2:
            print(f"[CV] skip fold {fold}: single class")
            continue
        base = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            random_state=SEED, class_weight="balanced_subsample", n_jobs=-1
        )
        method = "isotonic" if (y_tr.sum()>=200) else "sigmoid"
        model = CalibratedClassifierCV(base_estimator=base, method=method, cv=3)
        model.fit(X[tr], y_tr)
        p = model.predict_proba(X[va])[:,1]
        aucs.append(roc_auc_score(y_va, p))
        pr_aucs.append(average_precision_score(y_va, p))
        folds += 1

    if folds>0:
        print(f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (folds={folds})")
        print(f"PR-AUC: {np.mean(pr_aucs):.3f} ± {np.std(pr_aucs):.3f}")
    else:
        print("AUC/PR-AUC: n/a (all folds skipped)")

    # ====== 最終学習（全データ） ======
    base = RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=20,
        random_state=SEED, class_weight="balanced_subsample", n_jobs=-1
    )
    method = "isotonic" if (y.sum()>=300) else "sigmoid"
    final_model = CalibratedClassifierCV(base_estimator=base, method=method, cv=3)
    final_model.fit(X, y)

    joblib.dump({
        "model": final_model,
        "Xcols": Xcols,
        "meta": {
            "symbol": SYMBOL, "interval": INTERVAL,
            "H": H, "break_buffer": BREAK_BUFFER, "touch_buffer": TOUCH_BUFFER,
            "samples": int(len(ds)), "positives": int(y.sum()),
            "features": Xcols, "level_win": LEVEL_WIN, "level_stride": LEVEL_STRIDE
        }
    }, "models/break_model.joblib")
    print("saved -> models/break_model.joblib")

if __name__ == "__main__":
    main()
